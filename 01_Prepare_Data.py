# Databricks notebook source
# MAGIC %md The purpose of this notebook is to setup the data assets used in the Zingg Person Entity-Resolution solution accelerator. This notebook is available on https://github.com/databricks-industry-solutions/customer-er.

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC
# MAGIC In this notebook, we will download a dataset, break it up into records representing an initial data load and an incremental data load, and then persist the data to a database table. We also will take a moment to explore the data before proceeding to the entity resolution work taking place in the subsequent notebooks. 

# COMMAND ----------

# DBTITLE 1,Get Config
# MAGIC %run "./00_Intro_&_Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn

import os

# COMMAND ----------

# MAGIC %md ## Step 1: Reset the Environment
# MAGIC
# MAGIC Zingg depends on having the right data in just the right place.  To ensure we maintain a clean environment, we'll reset all the directories housing input, output and transient data.  In most environments, such a step should not be necessary.  This is just a precaution to ensure you get valid results should you run this series of notebooks multiple times:
# MAGIC
# MAGIC **NOTE** Running this step resets this solution accelerator.  Do not run the following code unless you are reinitializing the solution accelerator in its entirity.
# MAGIC
# MAGIC **NOTE** The cell below depends on you having already created the mount point discussed at the top of Step 2 within this notebook.

# COMMAND ----------

# DBTITLE 1,Reset the Directories
## CODE DISABLED TO ENABLE AUTOMATED TESTING

#for k,v in config['dir'].items(): # for each directory identified in config
#    dbutils.fs.rm(v, recurse=True)   # remove the dir and all child content
#    dbutils.fs.mkdirs(dir)           # recreate empty dir

# COMMAND ----------

# DBTITLE 1,Reset the Database
# delete database
_ = spark.sql('DROP DATABASE IF EXISTS {0} CASCADE'.format(config['database name']))

# create database to house mappings
_ = spark.sql('CREATE DATABASE IF NOT EXISTS {0}'.format(config['database name']))

# set database as default for queries
_ = spark.catalog.setCurrentDatabase(config['database name'] )

# COMMAND ----------

# MAGIC %md ## Step 2: Access Source Data
# MAGIC
# MAGIC For this solution accelerator, we'll make use of the [North Carolina Voters 5M](https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution) dataset made available by the [Database Group Leipzig](https://dbs.uni-leipzig.de/en). This dataset, more fully documented in [this paper](https://dbs.uni-leipzig.de/file/famer-adbis2017.pdf), contains name and limited address information for several million registered voters within the state of North Carolina. There are duplicate records purposefully inserted into the set with specific adjustments to make them fuzzy matchable, bringing the total number of records in the dataset to around 5-million and, hence, the name of the dataset.
# MAGIC
# MAGIC The dataset is made available for download as a gzipped TAR file which needs to be downloaded, unzipped, and untarred to a folder named *downloads* under a [mount point](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs) you will need to setup in advance in your environment. In our environment, we've used a default name of */mnt/zingg_ncvoters* for our mount point.  You can alter this in the *00* notebook if you've elected to create a mount point using a different name:

# COMMAND ----------

# DBTITLE 1,Make Downloads Folder Location Accessible to Shell Script
os.environ['DOWNLOADS_FOLDER'] = '/dbfs' + config['dir']['downloads']

# COMMAND ----------

# DBTITLE 1,Download Data Assets
# MAGIC %sh
# MAGIC
# MAGIC # move to the downloads folder
# MAGIC rm -rf $DOWNLOADS_FOLDER
# MAGIC mkdir -p $DOWNLOADS_FOLDER
# MAGIC cd /$DOWNLOADS_FOLDER
# MAGIC
# MAGIC # download the data file
# MAGIC wget -q https://dbs.uni-leipzig.de/ds/5Party-ocp20.tar.gz
# MAGIC
# MAGIC # decompress the data file
# MAGIC tar -xf 5Party-ocp20.tar.gz
# MAGIC mv ./5Party-ocp20/*.csv ./
# MAGIC rm 5Party-ocp20.tar.gz
# MAGIC rm -r 5Party-ocp20
# MAGIC
# MAGIC # list download folder contents
# MAGIC ls -l

# COMMAND ----------

# MAGIC %md The North Carolina Voters 5M dataset consists of 5 files containing roughly 1-million records each. We will use the data in the first 4 files as our initial dataset and the 5th file for our incremental dataset:

# COMMAND ----------

# DBTITLE 1,Verify File Count
# count the files in the downloads directory
file_count = len(dbutils.fs.ls(config['dir']['downloads']))

print('Expecting 5 files in {0}'.format(config['dir']['downloads']))
print('Found {0} files in {1}'.format(file_count, config['dir']['downloads']))

# COMMAND ----------

# DBTITLE 1,Move Raw Inputs into Initial & Incremental Folders
# function to help with file copy
def copy_file_with_overwrite(from_file_path, to_file_path):
  
  # remove to-file if already exists
  try: 
    dbutils.fs.rm(to_file_path)
  except:
    pass
  
  # copy from-file to intended destination
  dbutils.fs.cp(from_file_path, to_file_path)


# for each file in downloaded dataset
for file in dbutils.fs.ls(config['dir']['downloads']):
  
  # determine file number (ncvr_numrec_1000000_modrec_2_ocp_20_myp_<int>_nump_5.csv)
  file_num = int(file.name.split('_')[-3])
  
  # if 0 - 3: copy to initial folder
  if file_num < 4:
    copy_file_with_overwrite(file.path, config['dir']['input'] + '/initial/' + file.name)
    
  # if 4: copy to incremental folder
  elif file_num == 4:
    copy_file_with_overwrite(file.path, config['dir']['input'] + '/incremental/' + file.name)

# COMMAND ----------

# DBTITLE 1,Verify Initial Dataset Has 4 Files
display(
  dbutils.fs.ls(config['dir']['input'] + '/initial')
  )

# COMMAND ----------

# DBTITLE 1,Verify Incremental Dataset Has 1 File
display(
  dbutils.fs.ls(config['dir']['input'] + '/incremental')
  )

# COMMAND ----------

# MAGIC %md ##Step 3: Persist Data to Database Tables
# MAGIC
# MAGIC Next, we will move our initial and staging data into delta tables to enable easier access in later notebooks. It's important to note that to make use of a JAR-based library like Zingg within a [Unity Catalog](https://docs.databricks.com/data-governance/unity-catalog/index.html) enabled Databricks workspace, you will need to make these tables external tables at this point in time.  You will more clearly see why in the subsequent notebooks as we define our Zingg pipelines:

# COMMAND ----------

# DBTITLE 1,Define Logic to Write CSV Inputs to Delta Table
def csv_to_delta_table(
  dbfs_folder_path,
  table_name,
  make_external=False
  ):

  # read folder data to data frame
  df = (
    spark
      .read
      .csv(
        path=dbfs_folder_path,
        sep=',',
        header=True,
        inferSchema=True
        )
    )
  
  # establish base writing logic
  df_writer = df.write.format('delta').mode('overwrite').option('overwriteSchema','true')

  # make table external if required
  if make_external:
    df_writer = df_writer.option('path', f"{config['dir']['tables']}/{table_name}")
  
  # write data to table
  _ = df_writer.saveAsTable(table_name)

# COMMAND ----------

# DBTITLE 1,Write Initial Data to Delta Table
csv_to_delta_table(
  dbfs_folder_path=config['dir']['input'] + '/initial/', 
  table_name='initial',
  make_external=True
  )

display(
  spark.sql(f"DESCRIBE EXTENDED initial")
  )

# COMMAND ----------

# DBTITLE 1,Write Incremental Data to Delta Table
csv_to_delta_table(
  dbfs_folder_path=config['dir']['input'] + '/incremental/', 
  table_name='incremental',
  make_external=True
  )

display(
  spark.sql(f"DESCRIBE EXTENDED incremental")
  )

# COMMAND ----------

# MAGIC %md ## Step 4: Examine the Data
# MAGIC
# MAGIC To get a sense of the data, we'll examine the records in the initial dataset:

# COMMAND ----------

# DBTITLE 1,Access Initial Data Set
initial = (
  spark
    .table('initial')
  )

display(initial)

# COMMAND ----------

# MAGIC %md In the dataset, voters are identified based on the following fields:</p>
# MAGIC
# MAGIC * **givenname** - the first or *given* name of the person
# MAGIC * **surname** - the family or *surname* of the person
# MAGIC * **suburb** - the town, city or other municipal entity associated with the person
# MAGIC * **postcode** - the postal (zip) code associated with the person 
# MAGIC
# MAGIC A unique identifier, *recid*, is used in the original dataset to identify unique records.  This identifier enables us to validate some portion of our work but we will need to make sure Zingg ignores this field so that it does not learn that two rows with the same *recid* are a match.
# MAGIC
# MAGIC To create duplicates in the dataset, the team responsible for creating it simply re-inserted some number of the rows back into it without any modifications. Another set of duplicates was created by re-inserting rows while *corrupting* one or multiple of the 4 fields identified above.  Corruptions take the form of the removal, replacement or reversal of some number or characters from within a string as would be typical of a poor data entry process.  These duplicates are identifiable by their matching *recid* values:

# COMMAND ----------

# DBTITLE 1,Identify Author-Generated Duplicates
# rec ids of records with multiple entries
dups = (
  initial
    .select('recid')
    .groupBy('recid')
      .agg(fn.count('*').alias('recs'))
    .filter('recs>1')
    .select('recid')
  )

# displa full record for identified dups
display(
  initial
    .join(dups, on='recid')
    .orderBy('recid')
  )

# COMMAND ----------

# MAGIC %md Still other duplicates are naturally occuring in the dataset.  With a dataset of this size, it's not unexpected that some errors were not caught following data entry.  For example, consider these records which appear to be exact duplicates but which have separate *recid* values. It is possible that two individuals within a given zip code have the same first and last name so that some of these records only appear to be duplicates given the lack of additional identifying data in the dataset. However, the uniqueness of some of these names would indicate that some are true duplicates in the original dataset:

# COMMAND ----------

# DBTITLE 1,Identify Apparent Duplicates In Original Dataset
dups = (
  initial.alias('a')
    .join(
      initial.alias('b'), 
      on=['givenname','surname','suburb','postcode']
      )
    .filter('a.recid != b.recid')
    .select('a.recid','a.givenname','a.surname','a.suburb','a.postcode')
    .distinct()
    .orderBy('givenname','surname','suburb','postcode')
  )
  
display(dups)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | zingg                                  | entity resolution library | GNU Affero General Public License v3.0    | https://github.com/zinggAI/zingg/                       |
