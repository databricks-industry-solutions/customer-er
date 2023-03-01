# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/customer-er. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/customer-entity-resolution.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to demonstrate the initial workflow by which Zingg trains a model to be used for (later) incremental data processing.  

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC The initial Zingg workflow consists of two primary steps plus one additional step that is often performed to initialize the environment.  These three steps are:</p>
# MAGIC 
# MAGIC 1. Label Training Data
# MAGIC 2. Train Model on Labeled Data
# MAGIC 3. Perform Initial Deduplication
# MAGIC 
# MAGIC The end result of this workflow is a trained model and tables capturing which records in the initial dataset match with each other. 
# MAGIC 
# MAGIC Each of the three steps is facilitated by a separate job that was setup in the *ER Setup 02* notebook.  If you haven't successfully run that notebook (as well as the *ER Setup 01* notebook), please do so before proceeding with this one.

# COMMAND ----------

# DBTITLE 1,Install Required LIbraries
# MAGIC %pip install tabulate

# COMMAND ----------

# DBTITLE 1,Initialize Config
# MAGIC %run "./00.0_ Intro & Config"

# COMMAND ----------

# DBTITLE 1,Set Additional Configurations
# folders housing candidate pairs, labeled (marked) and unlabeled (unmarked)
MARKED_DIR = config['dir']['zingg'] + '/' + config['model name'] + '/trainingData/marked'
UNMARKED_DIR = config['dir']['zingg'] + '/' + config['model name'] + '/trainingData/unmarked'
OUTPUT_DIR = config['dir']['output'] + '/initial'

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pandas as pd
import numpy as np

import time
import uuid

from tabulate import tabulate

# COMMAND ----------

# MAGIC %md ##Step 0: Setup Helper Functions & Widget
# MAGIC 
# MAGIC As mentioned in the *ER Setup 00* notebook, Zingg provides building block components for the construction of an entity-resolution workflow application.  To keep things simple, we will attempt to emulate an application workflow from within this notebook, recognizing that most applications would provide users with a specialized client UI for the labeling and data interpretation work performed here.
# MAGIC 
# MAGIC To assist us in implementing an in-notebook experience, we'll define a few helper functions now:

# COMMAND ----------

# DBTITLE 1,Reset Label Stage
def reset_labels(reset_model=True):
  '''
  The purpose of this function is to reset the labeled data record in
  prior iterations of the label stage.  Because the Zingg model trained
  on these data is invalidated by this step, the reset_model argument
  instructs the function to delete the model information as well.
  '''
  
  # drop marked (labeled) data 
  dbutils.fs.rm(MARKED_DIR, recurse=True)
  
  # drop unmarked data
  dbutils.fs.rm(UNMARKED_DIR, recurse=True)
  
  if reset_model:
    dbutils.fs.rm('{0}/{1}'.format(config['dir']['zingg'], config['model name']), recurse=True)
    dbutils.fs.rm('{0}'.format(config['dir']['output']), recurse=True)
    
  return

# COMMAND ----------

# DBTITLE 1,Generate Candidate Pairs
# run the find training job
def run_find_training_job():
  '''
  The purpose of this function is to run the Zingg findTraining job that generates
  candidate pairs from the initial set of data specified in the job's configuration
  '''
  
  # identify the find training job
  find_training_job = ZinggJob( config['job']['initial']['findTrainingData'], config['job']['databricks workspace url'], config['job']['api token'])
  
  # run the job and wait for its completion
  find_training_job.run_and_wait()

  return


# retrieve candidate pairs
def get_candidate_pairs():
  '''
  The purpose of this function is to retrieve candidate pairs that need labeling.
  The function compares the content of the unmarked folder within which the Zingg
  findTraining job deposits candidate paris with those of the marked folder where
  we persist labeled pairs so that no previously labeled pairs are returned.
  '''
  unmarked_pd = pd.DataFrame({'z_cluster':[]})
  marked_pd = pd.DataFrame({'z_cluster':[]})
  
  # read unmarked pairs
  try:
    tmp_pd = pd.read_parquet(
        '/dbfs'+ UNMARKED_DIR, 
        engine='pyarrow'
         )
    if tmp_pd.shape[0] != 0: unmarked_pd = tmp_pd
  except:
    pass
  
  # read marked pairs
  try:
    tmp_pd = pd.read_parquet(
        '/dbfs'+ MARKED_DIR, 
        engine='pyarrow'
         )
    if tmp_pd.shape[0] != 0: marked_pd = tmp_pd
  except:
    pass
  
  # get unmarked not in marked
  candidate_pairs_pd = unmarked_pd[~unmarked_pd['z_cluster'].isin(marked_pd['z_cluster'])]
  
  return candidate_pairs_pd

# COMMAND ----------

# DBTITLE 1,Assign Labels to Candidate Pairs
# assign label to candidate pair
def assign_label(candidate_pairs_pd, z_cluster, label):
  '''
  The purpose of this function is to assign a label to a candidate pair
  identified by its z_cluster value.  Valid labels include:
     0 - not matched
     1 - matched
     2 - uncertain
  '''
  
  # assign label
  candidate_pairs_pd.loc[ candidate_pairs_pd['z_cluster']==z_cluster, 'z_isMatch'] = label
  
  return

# persist labels to marked folder
def save_labels(candidate_pairs_pd):
  '''
  The purpose of this function is to save labeled pairs to the unmarked folder.
  '''

  # make dir if not exists
  dbutils.fs.mkdirs(MARKED_DIR)

  # save labeled data to file
  candidate_pairs_pd.to_parquet(
   '/dbfs/' + MARKED_DIR +'/markedRecords_'+ str(time.time_ns()/1000) + '.parquet', 
    compression='snappy',
    index=False # do not include index
    )
  
  return


def count_labeled_pairs():
  '''
  The purpose of this function is to count the labeled pairs in the marked folder.
  '''
  
  # create initial dataframes
  marked_pd = pd.DataFrame({'z_cluster':[]})
  
  # read unmarked pairs
  try:
    marked_pd = pd.read_parquet(
        '/dbfs'+ MARKED_DIR, 
        engine='pyarrow'
         )
  except:
    pass
  
  n_total = len(np.unique(marked_pd['z_cluster']))
  n_positive = len(np.unique(marked_pd[marked_pd['z_isMatch']==1]['z_cluster']))
  n_negative = len(np.unique(marked_pd[marked_pd['z_isMatch']==0]['z_cluster']))
  
  return n_positive, n_negative, n_total

# COMMAND ----------

# DBTITLE 1,Setup DropDown Widget
# setup widget 
available_labels = {
    'No Match':0,
    'Match':1,
    'Uncertain':2
    }
dbutils.widgets.dropdown('label', 'Uncertain', available_labels.keys(), 'Is this pair a match?')

# COMMAND ----------

# MAGIC %md ##Step 1: Label Training Data
# MAGIC 
# MAGIC With our helper function in place, we can now implement the first step of the initial workflow.  Within this step, Zingg will read an initial set of input data and from it generate a set of record pairs that it believes may be duplicates.  As "expert data reviewers", we will review each pair and label it as either a *Match* or *No Match*.  (We may also label it as *Uncertain* if we cannot determine if the records are a match.)  The labeled data will be persisted to facilitate the model training step that follows.</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/er_findtrainingdata_dataflow.png' width=800>
# MAGIC 
# MAGIC The Zingg job called for this step is *zingg_initial_findTrainingData*.  This job uses a set of *blocking* techniques to identify potential duplicates in the initial dataset.  Some techniques work better than others so as you perform the multiple cycles of candidate pair generation and labeling required before we can proceed to model training, you will notice some candidate pairs suggestions are better than others.  It is common that the quality of the suggestions ebbs and flows as you move through various pair generation cycles.  Please understand that this is part of the process by which Zingg learns.

# COMMAND ----------

# DBTITLE 1,Reset Label Data (Optional)
# only enable this if you want to 
# delete all previously labeled data
if True: 
  reset_labels(reset_model=True)
  print('Labeled data deleted')

# COMMAND ----------

# MAGIC %md To get started, we generate our candidate pairs:
# MAGIC 
# MAGIC **NOTE** This step will trigger the *zingg_initial_findTrainingData* job only if no unlabeled pairs exist from prior runs. If unlabeled pairs exist from prior runs, the routine will return those to be labeled before triggering the job.  Because different algorithms are used to identify potential duplicates between runs, some instances of the will run noticeably longer than others.

# COMMAND ----------

# DBTITLE 1,Get Data (Run Once Per Cycle)
# get candidate pairs
candidate_pairs_pd = get_candidate_pairs()

# if no candidate pairs, run job and wait
if candidate_pairs_pd.shape[0] == 0:
  print('No unlabeled candidate pairs found.  Running findTraining job ...')
  run_find_training_job()
  candidate_pairs_pd = get_candidate_pairs()
  
# get list of pairs (as identified by z_cluster) to label 
z_clusters = list(np.unique(candidate_pairs_pd['z_cluster'])) 

# identify last reviewed cluster
last_z_cluster = '' # none yet

# print candidate pair stats
print('{0} candidate pairs found for labeling'.format(len(z_clusters)))

# COMMAND ----------

# MAGIC %md With candidate pairs now available for labeling, we are presented with one pair at a time and are tasked with using the drop-down widget at the top of this notebook to assign a label to each one.  As you consider each pair, remember that pairs with a shared *recid* values are definitely duplicates but some duplicates may exist for which the *recid* values differ. (Per our job configurations, the *recid* field is not used for record matching.)
# MAGIC 
# MAGIC Once the widget reflects your label assignment, re-run the cell to assign the label and bring up the next pair:
# MAGIC 
# MAGIC **NOTE** Changing the value of the widget will trigger the following cell to re-execute.  You can disable this functionality by clicking ion the settings icon in the widget bar and changing *On Widget Change* to *Do Nothing*.

# COMMAND ----------

# DBTITLE 1,Perform Labeling (Run Repeatedly Until All Candidate Pairs Labeled)
# get current label setting (which is from last cluster)
last_label = available_labels[dbutils.widgets.get('label')]

# assign label to last cluster
if last_z_cluster != '':
  assign_label(candidate_pairs_pd, last_z_cluster, last_label)

# get next cluster to label
try:
  z_cluster = candidate_pairs_pd[(candidate_pairs_pd['z_isMatch']==-1) & (candidate_pairs_pd['z_cluster'] != last_z_cluster)].head(1)['z_cluster'].values[0]
except:
  pass
  z_cluster = ''

# present the next pair
if z_cluster != '':
  print('IS THIS PAIR A MATCH?')
  print(f"Current widget setting will label this as '{dbutils.widgets.get('label')}'.")
  print('Change widget value if different label required.\n')
  print(
    tabulate(
      candidate_pairs_pd[candidate_pairs_pd['z_cluster']==z_cluster][['recid','givenname','surname','suburb','postcode']], 
      headers = 'keys', 
      tablefmt = 'psql'
      )
    )
else:
  print('All candidate pairs have been labeled.\n')

# hold last items for assignnment in next run
last_z_cluster = z_cluster

# if no more to label
if last_z_cluster == '':
  
  # save labels
  save_labels(candidate_pairs_pd)
  
  # count labels accumulated
  n_pos, n_neg, n_tot = count_labeled_pairs()
  print(f'You have accumulated {n_pos} pairs labeled as positive matches.')
  print("If you need more pairs to label, re-run the cell titled 'Get Data (Run Once Per Cycle).'")

# COMMAND ----------

# MAGIC %md You are encouraged to execute the block above repeatedly until all candidate pairs are labeled. Here we provide some marked data to make sure the labeling outcome is consistent between runs. Please remove the following block if you would like to use your own marked data to train your model.

# COMMAND ----------

dbutils.fs.rm(MARKED_DIR, True)
dbutils.fs.rm(UNMARKED_DIR, True)
dbutils.fs.cp("s3://db-gtm-industry-solutions/data/rcg/customer_er/data/marked/", MARKED_DIR, True)
dbutils.fs.cp("s3://db-gtm-industry-solutions/data/rcg/customer_er/data/unmarked/", UNMARKED_DIR, True)

# COMMAND ----------

# MAGIC %md Before moving on to the next phase, it's a good idea to review the labels assigned to the candidate pairs for errors:

# COMMAND ----------

# DBTITLE 1,Review Labeled Pairs
marked_pd = pd.read_parquet(
      '/dbfs'+ MARKED_DIR, 
      engine='pyarrow'
       )
  
display(marked_pd)

# COMMAND ----------

# MAGIC %md Should you have any mislabeled pairs, simply run the following with the appropriate substitutions for each pair you wish to correct:
# MAGIC 
# MAGIC ```
# MAGIC 
# MAGIC # set values here
# MAGIC z_cluster = 'Z_CLUSTER VALUE ASSOCIATED WITH PAIR TO RELABEL'
# MAGIC new_label = available_labels['VALUE FROM WIDGET DROP DOWN TO ASSIGN']
# MAGIC 
# MAGIC 
# MAGIC # read existing data
# MAGIC marked_pd = pd.read_parquet(
# MAGIC       '/dbfs'+ MARKED_DIR, 
# MAGIC       engine='pyarrow'
# MAGIC        )
# MAGIC 
# MAGIC # assign new label
# MAGIC marked_pd.loc[ marked_pd['z_cluster']==z_cluster, 'z_isMatch'] = label
# MAGIC 
# MAGIC # delete old records
# MAGIC dbutils.fs.rm(MARKED_DIR, recurse=True)
# MAGIC dbutils.fs.mkdirs(MARKED_DIR)
# MAGIC 
# MAGIC # write updated records
# MAGIC marked_pd.to_parquet(
# MAGIC    '/dbfs' + MARKED_DIR +'/markedRecords_'+ str(time.time_ns()/1000) + '.parquet', 
# MAGIC     compression='snappy',
# MAGIC     index=False # do not include index
# MAGIC     )
# MAGIC 
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## Step 2: Train Model on Labeled Data
# MAGIC 
# MAGIC To train the model against the labeled pairs, we simply kickoff the *zingg_initial_train* job which call's Zingg's *train* logic.  In this job, the labeled pairs are used to train a model which scores candidate pairs (generated by Zingg in later stages) for the probability of a match:</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/er_train_workflow2.png' width=500>

# COMMAND ----------

# DBTITLE 1,Train the Model
train_job = ZinggJob( config['job']['initial']['train'], config['job']['databricks workspace url'], config['job']['api token'])
train_job.run_and_wait()

# COMMAND ----------

# MAGIC %md ## Step 3: Perform Initial Deduplication
# MAGIC 
# MAGIC Using the trained model, we can examine the initial dataset to identify clusters of records.  A cluster is a group of records that are believed to be duplicates of one another.  The Zingg *match* logic combines both blocking and candidate pair scoring to generate the clustered results:</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/er_initial_match3.png' width=800>
# MAGIC 
# MAGIC **NOTE** Given the volume of initial data and the scale of the Databricks cluster assigned to the *match* job, this step may take a while to run.
# MAGIC 
# MAGIC Once the matched data are generated, we will capture the output to a set of tables that will enable incremental processing. This table structure is as follows:</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/er_schema.png' width=200>

# COMMAND ----------

# DBTITLE 1,Clear Output from Any Prior Runs
dbutils.fs.rm(OUTPUT_DIR, recurse=True)

# COMMAND ----------

# DBTITLE 1,Run Match Job
match_job = ZinggJob( config['job']['initial']['match'], config['job']['databricks workspace url'], config['job']['api token'])
match_job.run_and_wait()

# COMMAND ----------

# MAGIC %md Exploring the output of the match job, we can see how our model groups potentially duplicate records.  The output groups duplicates around a shared *z_cluster* value.  A min and max score, *i.e.* *z_minScore* and *z_maxScore*, is assigned to each cluster to help us understand the certainty with which records are assigned within a given cluster.  For clusters comprised of only one record, *i.e.* no duplicates were believed to have been found, these scores are omitted:

# COMMAND ----------

# DBTITLE 1,Review Matches
# retrieve initial matches
results = (
  spark
    .read
    .parquet(OUTPUT_DIR)
    .orderBy('z_cluster', ascending=True)
  )

# persist results to temp view
results.createOrReplaceTempView('matches')

# retrieve results from temp view
display(spark.table('matches'))

# COMMAND ----------

# MAGIC %md A quick spot check reveals that most clusters consist of 1 or 2 records.  However, there may be clusters with many more records based on how the duplicates were engineered for this dataset.  To get a quick sense of the cohesion of our clusters, we might take a quick look at the number of clusters associated with different counts of records:

# COMMAND ----------

# DBTITLE 1,Examine Number of Clusters by Record Count
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   record_count,
# MAGIC   COUNT(z_cluster) as clusters
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     z_cluster,
# MAGIC     count(*) as record_count
# MAGIC   FROM matches
# MAGIC   GROUP BY z_cluster
# MAGIC   ) 
# MAGIC GROUP BY record_count
# MAGIC ORDER BY record_count

# COMMAND ----------

# MAGIC %md A review the number of clusters by cluster member count shows that the majority of our clusters are believed to contain just a few duplicate records. However,  there are quite a few clusters with a large number of records associated with them.  It might be worth examining these to understand what may be happening here but it's important to keep in mind that we never expect perfection from our model.  If we feel our model could be better at defining clusters, it's important we return to the labeling phase of our work and then re-train and re-match our data.
# MAGIC 
# MAGIC With that in mind, it would be helpful if Zingg provided some high-level metrics and diagnostics to help us understand the performance of our model.  The reality is that outside of evaluation scenarios where we may have some form of ground-truth against to evaluate our results, its very difficult to clearly identify the precision of a model such as this.  Quite often, the best we can do is review the results and make a judgement call based on the volume of identifiable errors and the patterns associated with those errors to develop a sense of whether our model's performance is adequate for our needs.

# COMMAND ----------

# MAGIC %md If we are satisfied with our results, we can capture our cluster data to a table structure that will allow us to more easily perform incremental data processing. 

# COMMAND ----------

# DBTITLE 1,Create Table Structures
# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS clusters;
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS clusters (
# MAGIC   cluster_id bigint GENERATED ALWAYS AS IDENTITY,
# MAGIC   z_cluster string
# MAGIC   )
# MAGIC   USING DELTA;
# MAGIC   
# MAGIC DROP TABLE IF EXISTS cluster_members;
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS cluster_members (
# MAGIC   cluster_id integer,
# MAGIC   givenname string,
# MAGIC   surname string,
# MAGIC   suburb string,
# MAGIC   postcode string
# MAGIC   )
# MAGIC   USING DELTA;

# COMMAND ----------

# MAGIC %md Now we can insert our cluster data into the appropriate table.  Note that we are modifying the integer-based cluster identifier generated during the matching process to be a string with an appended unique identifier.  If we chose to re-run the match step, the *z_cluster* values generated will overlap with those generated in prior runs.  If we wish to hold-on to some prior cluster assignments, it might be helpful to distinguish between clusters generated in different *match* cycles:

# COMMAND ----------

# DBTITLE 1,Get Unique Identifier
# create a unique identifier
guid = str(uuid.uuid4())
print(f"A unique identifier of '{guid}' will be assigned to clusters from this run.")

# COMMAND ----------

# DBTITLE 1,Insert Clusters Data
_ = spark.sql(f"""
INSERT INTO clusters (z_cluster)
SELECT DISTINCT 
  CONCAT('{guid}',':',CAST(z_cluster as string))
FROM matches
""")

display(spark.table('clusters'))

# COMMAND ----------

# MAGIC %md And now we can insert the cluster members:

# COMMAND ----------

# DBTITLE 1,Insert Record-to-Cluster Mapping Table
_ = spark.sql(f"""
INSERT INTO cluster_members (cluster_id, givenname, surname, suburb, postcode)
SELECT DISTINCT
  a.cluster_id,
  b.givenname,
  b.surname,
  b.suburb,
  b.postcode
FROM clusters a
INNER JOIN matches b
  ON a.z_cluster=CONCAT('{guid}',':',CAST(b.z_cluster as string))
""")

display(spark.table('cluster_members').orderBy('cluster_id'))

# COMMAND ----------

# MAGIC %md Of course, we may have some clusters we might want to manually correct.  Using standard DELETE, UPDATE and INSERT statements, we can update the delta lake formatted tables in this environment to achieve the results we require.  If we create new clusters to which to assign users as part of a manual correction, we might create a new entry in the *clusters* table by simply inserting a value as follows:
# MAGIC 
# MAGIC ```
# MAGIC import uuid
# MAGIC 
# MAGIC # create new entry
# MAGIC guid= str(uuid.uuid4())
# MAGIC _ = spark.sql(f"INSERT INTO clusters (z_cluster) VALUES('{guid}')")
# MAGIC 
# MAGIC # get cluster id for new entry
# MAGIC cluster_id = spark.sql(f"SELECT cluster_id FROM clusters WHERE z_cluster='{guid}'").collect()[0]['cluster_id']
# MAGIC print(f"New z_cluster '{guid}' created with cluster_id of {cluster_id}")
# MAGIC 
# MAGIC ```

# COMMAND ----------

# MAGIC %md But what happens if we decide to retrain our model after we've setup these mappings? As mentioned above, retraining our model will cause new clusters with overlapping integer *z_cluster* identifiers to be created.  In this scenario, you need to decide whether you wish to preserve any manually adjusted mappings from before or otherwise start over from scratch.  If starting over, then simply drop and recreate the *clusters* and *cluster_members* tables. If preserving manually adjusted records, the GUID value associated with each cluster will keep the cluster identifiers unique.  You'll need to then decide how records assigned to clusters by the newly re-trained model should be merged with the preserved cluster data. It's a bit of a juggle so this isn't something you'll want to do on a regular basis.

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
