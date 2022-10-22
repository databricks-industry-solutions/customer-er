# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/customer-er. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/customer-entity-resolution.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to demonstrate the incremental workflow by which Zingg applies a trained model to newly arriving data. 

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC The incremental Zingg workflow consists of two steps, each of which is intended to examine incoming data for the inclusion of duplicate records.  These steps are:</p>
# MAGIC 
# MAGIC 1. Identify duplicates between incoming and previously observed records (linking)
# MAGIC 2. Identify duplicates within the incoming dataset (matching)
# MAGIC 
# MAGIC As part of this workflow, newly observed linked and matched records are appended to the set of previously observed data.  This updated dataset then forms the basis for the next incremental update cycle.

# COMMAND ----------

# DBTITLE 1,Initialize Config
# MAGIC %run "./00.0_ Intro & Config"

# COMMAND ----------

# MAGIC %md **NOTE** This workflow is highly dependent upon data residing in the folder locations specified in the configuration files associated with the relevant Zingg jobs.  If you change the following values, be sure to update the corresponding values in the appropriate configuration files.

# COMMAND ----------

# DBTITLE 1,Set Additional Configurations
INCOMING_INPUT_DIR = config['dir']['input'] + '/incremental/incoming'
PRIOR_INPUT_DIR = config['dir']['input'] + '/incremental/prior'

LINK_OUTPUT_DIR = config['dir']['output'] + '/incremental/link'
MATCH_OUTPUT_DIR = config['dir']['output'] + '/incremental/match'

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pandas as pd
import uuid
import pyspark.sql.functions as fn

# COMMAND ----------

# MAGIC %md ## Step 0: Setup Required Data
# MAGIC 
# MAGIC The incremental workflow is dependent upon the arrival of new records needing deduplication.  To simulate this, we withheld one of the 5 data files in the downloaded dataset and split it into a smaller number of files with roughly 10,000 records in each. These files are currently housed in the staging folder:

# COMMAND ----------

# DBTITLE 1,Examine Staging Folder Contents
display(dbutils.fs.ls(config['dir']['staging']))

# COMMAND ----------

# MAGIC %md To simulate the 'arrival' of one of these files, we'll simply move it into an incremental input folder:

# COMMAND ----------

# DBTITLE 1,Moving Incoming Data into Input Folders
# delete any previous incremental data
dbutils.fs.rm(INCOMING_INPUT_DIR, recurse=True)

# get name of next incremental file
for incr_file in dbutils.fs.ls(config['dir']['staging']):
  if incr_file.size > 0: # skip subdirectories
      break

# copy file into position
dbutils.fs.cp(incr_file.path, INCOMING_INPUT_DIR + '/' + incr_file.name)

# archive this incremental file
arch_file_name = config['dir']['staging'] + '/archived/' + incr_file.name
try:
  dbutils.fs.rm(arch_file_name)
except:
  pass

dbutils.fs.mv(incr_file.path, arch_file_name)

# display incoming data folder contents
display(dbutils.fs.ls(INCOMING_INPUT_DIR))

# COMMAND ----------

# MAGIC %md In addition to having access to incoming data, the linking step of this workflow requires access to data previously observed.  We have been housing this data inside of a set of tables in a Databricks catalog.  As this data may be modified through user activity, we will need to export this data to our input folder at the start of each incremental run:
# MAGIC 
# MAGIC **NOTE** The link and match Zingg job's configuration files specify that an integer *recid* is present (but not used) in the incoming data.  We did not capture this field as part of the *cluster_members* table.  To meet the schema requirements, we'll put a dummy value of 0 in place of the *recid* as we setup the prior dataset.

# COMMAND ----------

# DBTITLE 1,Move Prior Data into Link Input Folder
# clean up the prior input folder
dbutils.fs.rm(PRIOR_INPUT_DIR, recurse=True)

# write latest version of priors to folder as csv (per specs in config file)
_ = (
  spark
    .table('cluster_members')
    .selectExpr('0 as recid','givenname','surname','suburb','postcode')
    .write
      .csv(
        path=PRIOR_INPUT_DIR,
        mode='overwrite',
        sep=',',
        header=True
        )
  )

# COMMAND ----------

# MAGIC %md ## Step 1: Identify Duplicates Between Incoming & Previous Data
# MAGIC 
# MAGIC With the required data in position, we can now link our incoming data to our priors by calling the appropriate Zingg job. The *link* logic executed through the *zingg_incremental_link* job compares the incoming and prior data to identify matches between the two using the model trained in the initial workflow:</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/er_incremental_link2.png' width=900>

# COMMAND ----------

# DBTITLE 1,Link Incoming to Prior Data
link_job = ZinggJob( config['job']['incremental']['link'], config['job']['databricks workspace url'], config['job']['api token'])
link_job.run_and_wait()

# COMMAND ----------

# MAGIC %md The result of the link job is an output dataset identifying the records in the incoming data that are likely to match those in the prior data:
# MAGIC 
# MAGIC **NOTE** With only 10,000 records per incremental file, it is possible that no linkages were detected between the incoming data and priors.
# MAGIC 
# MAGIC **NOTE** We are coalescing all null values to empty strings to make comparisons in the code that follows easier to implement.

# COMMAND ----------

# DBTITLE 1,Review Link Job Output
linked = (
  spark
    .read
    .parquet(LINK_OUTPUT_DIR)
    .selectExpr(
      "COALESCE(givenname,'') as givenname",
      "COALESCE(surname,'') as surname",
      "COALESCE(suburb,'') as suburb",
      "COALESCE(postcode,'') as postcode",
      'z_score',
      'z_cluster',
      'z_source'
      )
  )

display(linked.orderBy('z_cluster'))

# COMMAND ----------

# MAGIC %md The link job output assigns a *z_cluster* value to records in the incoming dataset likely to match a record in the prior dataset.  A *z_score* helps us understand the probability of that match. The *z_source* field differentiates between records coming from the prior and the incoming datasets.
# MAGIC 
# MAGIC It's important to note that an incoming record may be linked to more than one prior records. Also, incoming records that do not have likely matches in the prior dataset (as determined by the blocking portion of the Zingg logic), will not appear in the linking output.  This knowledge needs to be taken into the data processing steps that follow.
# MAGIC 
# MAGIC To help us work with the linked data, we might separate those records from the prior dataset from those in the incoming dataset.  For the prior dataset, we can lookup the *cluster_id* in our *cluster_members* table to make the appending of new data to that table easier in later steps:

# COMMAND ----------

# DBTITLE 1,Get Linked Priors
linked_prior = (
  linked
    .alias('a')
    .filter(fn.expr("a.z_source = 'input_incremental_link_prior'"))
    .join( 
      spark.table('cluster_members').alias('b'), 
      on=fn.expr("""
      a.givenname=COALESCE(b.givenname,'') AND 
      a.surname=COALESCE(b.surname,'') AND 
      a.suburb=COALESCE(b.suburb,'') AND 
      a.postcode=COALESCE(b.postcode,'')
      """)      
      )
    .selectExpr(
      'a.givenname',
      'a.surname',
      'a.suburb',
      'a.postcode',
      'b.cluster_id',
      'a.z_cluster',
      'a.z_score',
      'a.z_source'
      )
  )
linked_prior.createOrReplaceTempView('linked_prior')

# COMMAND ----------

# DBTITLE 1,Get Linked Incoming
linked_incoming = linked.filter(fn.expr("z_source = 'input_incremental_link_incoming'"))
linked_incoming.createOrReplaceTempView('linked_incoming')

# COMMAND ----------

# MAGIC %md We can now handle this data through the following *actions*:</p>
# MAGIC 
# MAGIC 1. For those prior records linked to incoming records with a probability above a given threshold, add the record to the *cluster_members* table (assuming the record is not identical to the one already in the table).
# MAGIC 2. For any priors assigned a linked incoming record not addressed in the prior step, hand those records over for expert review.

# COMMAND ----------

# DBTITLE 1,Action 1: Accept Good, High Scoring Linkages
# MAGIC %sql
# MAGIC 
# MAGIC INSERT INTO cluster_members (cluster_id, givenname, surname, suburb, postcode)
# MAGIC SELECT -- cluster assignment for these records
# MAGIC   o.cluster_id,
# MAGIC   m.givenname,
# MAGIC   m.surname,
# MAGIC   m.suburb,
# MAGIC   m.postcode
# MAGIC FROM ( -- matching records with this high score
# MAGIC   SELECT
# MAGIC     x.givenname,
# MAGIC     x.surname,
# MAGIC     x.suburb,
# MAGIC     x.postcode,
# MAGIC     x.z_score
# MAGIC   FROM linked_incoming x
# MAGIC   INNER JOIN (
# MAGIC     SELECT -- highest scoring record for each unique entry
# MAGIC       givenname,
# MAGIC       surname,
# MAGIC       suburb,
# MAGIC       postcode,
# MAGIC       max(z_score) as max_z_score
# MAGIC     FROM linked_incoming
# MAGIC     WHERE z_score >= 0.90 -- threshold
# MAGIC     GROUP BY givenname, surname, suburb, postcode
# MAGIC     ) y
# MAGIC     ON x.givenname=y.givenname AND x.surname=y.surname AND x.suburb=y.suburb AND x.postcode=y.postcode AND x.z_score=y.max_z_score
# MAGIC   GROUP BY 
# MAGIC     x.givenname,
# MAGIC     x.surname,
# MAGIC     x.suburb,
# MAGIC     x.postcode,
# MAGIC     x.z_score
# MAGIC   HAVING COUNT(*) = 1  -- make sure only one record matches high score
# MAGIC   ) m
# MAGIC INNER JOIN linked_incoming n -- locate the record and find its z_cluster value
# MAGIC   ON m.givenname=n.givenname AND m.surname=n.surname AND m.suburb=n.suburb AND m.postcode=n.postcode AND m.z_score=n.z_score
# MAGIC INNER JOIN linked_prior o -- find the prior record for this z_cluster
# MAGIC   ON n.z_cluster=o.z_cluster
# MAGIC WHERE -- something in the incoming record is different from what's already in prior
# MAGIC   m.givenname != o.givenname OR
# MAGIC   m.surname != o.surname OR
# MAGIC   m.suburb != o.suburb OR
# MAGIC   m.postcode != o.postcode

# COMMAND ----------

# DBTITLE 1,Action 2: Apply Expert Review to Remaining Linkages
# MAGIC %sql
# MAGIC 
# MAGIC WITH linked_incoming_not_in_cluster_members AS (
# MAGIC   SELECT
# MAGIC     a.z_cluster, 
# MAGIC     a.givenname,
# MAGIC     a.surname,
# MAGIC     a.suburb,
# MAGIC     a.postcode,
# MAGIC     a.z_source,
# MAGIC     a.z_score
# MAGIC   FROM linked_incoming a
# MAGIC   LEFT OUTER JOIN cluster_members b
# MAGIC     ON a.givenname=COALESCE(b.givenname,'') AND a.surname=COALESCE(b.surname,'') AND a.suburb=COALESCE(b.suburb,'') AND a.postcode=COALESCE(b.postcode,'')
# MAGIC   WHERE b.cluster_id Is Null
# MAGIC   )
# MAGIC SELECT * FROM linked_incoming_not_in_cluster_members
# MAGIC UNION ALL
# MAGIC SELECT 
# MAGIC   z_cluster, 
# MAGIC   givenname, 
# MAGIC   surname, 
# MAGIC   suburb, 
# MAGIC   postcode,
# MAGIC   z_source,
# MAGIC   z_score
# MAGIC FROM linked_prior
# MAGIC WHERE 
# MAGIC   z_cluster IN (SELECT z_cluster FROM linked_incoming_not_in_cluster_members)
# MAGIC ORDER BY z_cluster, z_source DESC

# COMMAND ----------

# MAGIC %md ## Step 2: Identify Duplicates within Incoming Data
# MAGIC 
# MAGIC The *link* logic identifies matches between the incoming and the prior data. Records not linked to prior data can be assumed to not be in the prior dataset.  But before simply inserting these new records into the database tables, we need to identify any duplicates in the incoming records themselves.  This is handled by applying the *match* logic to the incoming data.</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/er_incremental_match2.png' width=800>

# COMMAND ----------

# DBTITLE 1,Match Records within Incoming Data
match_job = ZinggJob( config['job']['incremental']['match'], config['job']['databricks workspace url'], config['job']['api token'])
match_job.run_and_wait()

# COMMAND ----------

# DBTITLE 1,Review Clusters
# retrieve matches
matches = (
  spark
    .read
    .parquet(MATCH_OUTPUT_DIR)
    .orderBy('z_cluster', ascending=True)
  )

# persist results to temp view
matches.createOrReplaceTempView('matches')

# retrieve results from temp view
display(spark.table('matches'))

# COMMAND ----------

# MAGIC %md With clusters assigned within the dataset, we can break down out data persistence actions as follows:</p>
# MAGIC 
# MAGIC 1. If a matching cluster has no records already in *cluster_members*, we can simply insert these clusters into the *clusters* and *cluster_members* tables.
# MAGIC 2. If a matching cluster has records already in *cluster_members* where there is linkage to one and only one cluster as recorded in the *clusters* table, then we can move all the matching cluster members under that cluster id.
# MAGIC 3. Any remaining matching clusters will require a manual review.

# COMMAND ----------

# DBTITLE 1,Get Unique Identifier
# create a unique identifier
guid = str(uuid.uuid4())
print(f"A unique identifier of '{guid}' will be assigned to clusters from this run.")

# COMMAND ----------

# DBTITLE 1,Action 1: No Linkage Exists
_ = ( # insert cluster records
  spark
    .sql(
      f"""
        INSERT INTO clusters (z_cluster)
        WITH z_clusters_with_records_in_cluster_members AS (
          SELECT DISTINCT
            a.z_cluster
          FROM matches a
          INNER JOIN cluster_members b
            ON 
              COALESCE(a.givenname,'')=COALESCE(b.givenname,'') AND
              COALESCE(a.surname,'')=COALESCE(b.surname,'') AND
              COALESCE(a.suburb,'')=COALESCE(b.suburb,'') AND
              COALESCE(a.postcode,'')=COALESCE(b.postcode,'')
          )
        SELECT DISTINCT 
          CONCAT('{guid}',':', CAST(z_cluster as string))
        FROM matches 
        WHERE 
          z_cluster NOT IN (
            SELECT z_cluster FROM z_clusters_with_records_in_cluster_members
            ) AND
          CONCAT('{guid}',':', CAST(z_cluster as string)) NOT IN (SELECT z_cluster FROM clusters) -- avoid repeat inserts
      """
    )
  )
  
_ = ( # insert cluster members
  spark
    .sql(
      f"""
      INSERT INTO cluster_members (cluster_id, givenname, surname, suburb, postcode)
      WITH z_clusters_with_records_in_cluster_members AS (
          SELECT DISTINCT
            a.z_cluster
          FROM matches a
          INNER JOIN cluster_members b
            ON 
              COALESCE(a.givenname,'')=COALESCE(b.givenname,'') AND
              COALESCE(a.surname,'')=COALESCE(b.surname,'') AND
              COALESCE(a.suburb,'')=COALESCE(b.suburb,'') AND
              COALESCE(a.postcode,'')=COALESCE(b.postcode,'')
          )
        SELECT p.*
        FROM (
          SELECT DISTINCT
            n.cluster_id,
            m.givenname,
            m.surname,
            m.suburb,
            m.postcode
          FROM matches m
          INNER JOIN (
            SELECT DISTINCT 
              y.cluster_id,
              x.z_cluster
            FROM matches x
            INNER JOIN clusters y
              ON CONCAT('{guid}',':', CAST(x.z_cluster as string))=y.z_cluster
            WHERE 
              x.z_cluster NOT IN (
                SELECT z_cluster FROM z_clusters_with_records_in_cluster_members
                )
              ) n
              ON m.z_cluster=n.z_cluster
            ) p
          LEFT OUTER JOIN cluster_members q -- avoid repeat inserts
            ON 
              COALESCE(p.givenname,'')=COALESCE(q.givenname,'') AND
              COALESCE(p.surname,'')=COALESCE(q.surname,'') AND
              COALESCE(p.suburb,'')=COALESCE(q.suburb,'') AND
              COALESCE(p.postcode,'')=COALESCE(q.postcode,'')
          WHERE q.cluster_id Is Null
      """
    )
  )

# COMMAND ----------

# DBTITLE 1,Action 2: Linkage through Match Cluster
# MAGIC %sql
# MAGIC 
# MAGIC INSERT INTO cluster_members (cluster_id, givenname, surname, suburb, postcode)
# MAGIC WITH z_clusters_with_linkage (
# MAGIC     SELECT DISTINCT
# MAGIC       a.z_cluster,
# MAGIC       b.cluster_id
# MAGIC     FROM matches a
# MAGIC     INNER JOIN cluster_members b
# MAGIC       ON 
# MAGIC         COALESCE(a.givenname,'')=COALESCE(b.givenname,'') AND
# MAGIC         COALESCE(a.surname,'')=COALESCE(b.surname,'') AND
# MAGIC         COALESCE(a.suburb,'')=COALESCE(b.suburb,'') AND
# MAGIC         COALESCE(a.postcode,'')=COALESCE(b.postcode,'')
# MAGIC     ),
# MAGIC   z_clusters_only_one_cluster_linkage AS (
# MAGIC     SELECT
# MAGIC       z_cluster
# MAGIC     FROM z_clusters_with_linkage
# MAGIC     GROUP BY z_cluster
# MAGIC     HAVING COUNT(*)=1
# MAGIC     )
# MAGIC SELECT DISTINCT
# MAGIC   y.cluster_id,
# MAGIC   x.givenname,
# MAGIC   x.surname,
# MAGIC   x.suburb,
# MAGIC   x.postcode
# MAGIC FROM matches x
# MAGIC INNER JOIN z_clusters_with_linkage y
# MAGIC   ON x.z_cluster=y.z_cluster
# MAGIC LEFT OUTER JOIN cluster_members z 
# MAGIC   ON 
# MAGIC     y.cluster_id=z.cluster_id AND
# MAGIC     COALESCE(x.givenname,'')=COALESCE(z.givenname,'') AND
# MAGIC     COALESCE(x.surname,'')=COALESCE(z.surname,'') AND
# MAGIC     COALESCE(x.suburb,'')=COALESCE(z.suburb,'') AND
# MAGIC     COALESCE(x.postcode,'')=COALESCE(z.postcode,'')
# MAGIC WHERE 
# MAGIC   x.z_cluster IN (SELECT z_cluster FROM z_clusters_only_one_cluster_linkage) AND
# MAGIC   z.cluster_id Is Null

# COMMAND ----------

# DBTITLE 1,Action 3: Matches Requiring Manual Review
# MAGIC %sql
# MAGIC 
# MAGIC WITH z_clusters_with_records_NOT_in_cluster_members AS (
# MAGIC   SELECT DISTINCT
# MAGIC     a.z_cluster
# MAGIC   FROM matches a
# MAGIC   LEFT OUTER JOIN cluster_members b
# MAGIC     ON 
# MAGIC       COALESCE(a.givenname,'')=COALESCE(b.givenname,'') AND
# MAGIC       COALESCE(a.surname,'')=COALESCE(b.surname,'') AND
# MAGIC       COALESCE(a.suburb,'')=COALESCE(b.suburb,'') AND
# MAGIC       COALESCE(a.postcode,'')=COALESCE(b.postcode,'')
# MAGIC   WHERE
# MAGIC     b.cluster_id Is Null
# MAGIC   )
# MAGIC SELECT *
# MAGIC FROM matches 
# MAGIC WHERE
# MAGIC   z_cluster IN (SELECT z_cluster FROM z_clusters_with_records_NOT_in_cluster_members)

# COMMAND ----------

# MAGIC %md With the incremental data processed and the *cluster_members* table updated, we are ready to repeat this workflow for the next incoming set of data.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | zingg                                  | entity resolution library | GNU Affero General Public License v3.0    | https://github.com/zinggAI/zingg/                       |
# MAGIC | tabulate | pretty-print tabular data in Python | MIT License | https://pypi.org/project/tabulate/ |
# MAGIC | filesplit | Python module that is capable of splitting files and merging it back | MIT License | https://pypi.org/project/filesplit/ |
