# Databricks notebook source
# MAGIC %md The purpose of this notebook is to train a Zingg model on labeled data as part of the the initial workflow in the Zingg Person Entity-Resolution solution accelerator. This notebook is available on https://github.com/databricks-industry-solutions/customer-er.
# MAGIC

# COMMAND ----------

# MAGIC %md **IMPORTANT NOTE** If you have just run Part A of the initial workflow, be sure to restart your Databricks cluster to avoid a potential error when attempting to run the steps below.

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC
# MAGIC The purpose of this notebook is to train a Zingg model using an initial dataset within which we know there are some duplicate records. Those duplicates were labeled in Part A of this initial workflow.  In Part B, we will use these data to perform two tasks:</p>
# MAGIC
# MAGIC 1. **train** - train a Zingg model using the labeled pairs resulting from user interactions on data coming from (multiple iterations of) the findTrainingData task.
# MAGIC 2. **match** - use the trained Zingg model to match duplicate records in the initial dataset.
# MAGIC </p>
# MAGIC
# MAGIC **NOTE** Because *train* and *match* are so frequently run together, Zingg provides a combined *trainMatch* task that we will employ to minimize the time to complete our work. We will continue to speak of these as two separate tasks but they will be executed as a combined task in this notebook.
# MAGIC
# MAGIC While the *train* task does not write any explicit output, it does persist trained model assets to the *zingg* model folder.  The *findTrainingData* task will also send intermidiary data outputs to an *unmarked* folder under the *zingg* model folder.  (It will also expect us to place some data in a *marked* frolder location under that same *zingg* model folder.) Understanding where these assets reside can be helpful should you need to troubleshoot certain issues you might encounter during the *findTrainingData* or *train* tasks.
# MAGIC
# MAGIC Before jumping into these steps, we first need to verify the Zingg application JAR is properly installed on this cluster and then install the Zingg Python wrapper which provides a simple API for using the application:

# COMMAND ----------

# DBTITLE 1,Verify Zingg JAR Installed
# set default zingg path
zingg_jar_path = None

# for each jar in the jars folder
for j in dbutils.fs.ls('/FileStore/jars'):
  # locate the zingg jar
  if '-zingg_' in j.path:
    zingg_jar_path = j.path
    print(f'Zingg JAR found at {zingg_jar_path}')
    break
    
if zingg_jar_path is None: 
  raise Exception('The Zingg JAR was NOT found.  Please install the JAR file before proceeding.')

# COMMAND ----------

# DBTITLE 1,Install Zingg Python Library
# MAGIC %pip install zingg

# COMMAND ----------

# DBTITLE 1,Initialize Config
# MAGIC %run "./00_Intro_&_Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pandas as pd
import numpy as np

import time
import uuid

from zingg.client import Arguments, ClientOptions, ZinggWithSpark
from zingg.pipes import Pipe, FieldDefinition, MatchType

from ipywidgets import widgets, interact

import pyspark.sql.functions as fn

# COMMAND ----------

# MAGIC %md ##Step 1: Configure the Zingg Client
# MAGIC
# MAGIC With Zingg's new [Python API](https://docs.zingg.ai/zingg/working-with-python), we can configure the Zingg tasks to read data from a given input data set, perform matching based on logic assigned to each field in the dataset, and generate output results (as part of the *match* task) in a specific format and structure.  These inputs are captured as a collection of arguments, the first of which we will assign being those associated with the model and its folder location:
# MAGIC
# MAGIC **NOTE** The client configuration steps presented here are identical to those in Part A.

# COMMAND ----------

# DBTITLE 1,Initialize Zingg Arguments
args = Arguments()

# COMMAND ----------

# DBTITLE 1,Assign Model Arguments

# this is where zingg models and intermediary assets will be stored
args.setZinggDir(config['dir']['zingg'] )

# this uniquely identifies the model you are training
args.setModelId(config['model name'])

# COMMAND ----------

# MAGIC %md Our next arguments are the input and output [pipes](https://docs.zingg.ai/zingg/connectors/pipes) which define where and in what format  data is read or written.
# MAGIC
# MAGIC For our input pipe, we are reading from a table in the [delta lake format](https://delta.io/).  Because this format captures schema information, we do not need to provide any additional structural details.  However, because Zingg doesn't have the ability to read this as a table in a Unity Catalog-enabled Databricks workspace, we've implemented the input table as an [external table](https://docs.databricks.com/sql/language-manual/sql-ref-external-tables.html) and are pointing our input pipe to the location where that table houses its data:

# COMMAND ----------

# DBTITLE 1,Config Model Inputs
# get location of initial table's data
input_path = spark.sql("DESCRIBE DETAIL initial").select('location').collect()[0]['location']

# configure Zingg input pipe
inputPipe = Pipe(name='initial', format='delta')
inputPipe.addProperty('path', input_path )

# add input pipe to arguments collection
args.setData(inputPipe)

# COMMAND ----------

# DBTITLE 1,Config Model Outputs
output_dir = config['dir']['output'] + '/initial'

# configure Zingg output pipe
outputPipe = Pipe(name='initial_matched', format='delta')
outputPipe.addProperty('path', output_dir)

# add output pipe to arguments collection
args.setOutput(outputPipe)

# COMMAND ----------

# MAGIC %md Next, we need to define how each field from our input pipe will be used by Zingg.  This is what Zingg refers to as a [field definition](https://docs.zingg.ai/zingg/stepbystep/configuration/field-definitions). The logic accessible to Zingg depends on the Zingg MatchType assigned to each field.  The MatchTypes supported by Zingg at the time this notebook was developed were:
# MAGIC </p>
# MAGIC
# MAGIC * **DONT_USE** - appears in the output but no computation is done on these
# MAGIC * **EMAIL** - matches only the id part of the email before the @ character
# MAGIC * **EXACT** - no tolerance with variations, Preferable for country codes, pin codes, and other categorical variables where you expect no variations
# MAGIC * **FUZZY** - broad matches with typos, abbreviations, and other variations
# MAGIC * **NULL_OR_BLANK** - by default Zingg marks matches as
# MAGIC * **NUMERIC** - extracts numbers from strings and compares how many of them are same across both strings
# MAGIC * **NUMERIC_WITH_UNITS** - extracts product codes or numbers with units, for example 16gb from strings and compares how many are same across both strings
# MAGIC * **ONLY_ALPHABETS_EXACT** - only looks at the alphabetical characters and compares if they are exactly the same
# MAGIC * **ONLY_ALPHABETS_FUZZY** - ignores any numbers in the strings and then does a fuzzy comparison
# MAGIC * **PINCODE** - matches pin codes like xxxxx-xxxx with xxxxx
# MAGIC * **TEXT** - compares words overlap between two strings
# MAGIC
# MAGIC For our needs, we'll make use of fuzzy matching on most of our fields while instructing Zingg to ignore the *recid* field as discussed in notebook *01*:

# COMMAND ----------

# DBTITLE 1,Configure Field Definitions
# define logic for each field in incoming dataset
recid = FieldDefinition('recid', 'integer', MatchType.DONT_USE)
givenname = FieldDefinition("givenname", 'string', MatchType.FUZZY)
surname = FieldDefinition('surname', 'string', MatchType.FUZZY)
suburb = FieldDefinition('suburb', 'string', MatchType.FUZZY)
postcode = FieldDefinition('postcode', 'string', MatchType.FUZZY)

# define sequence of fields to receive
field_defs = [recid, givenname, surname, suburb, postcode]

# add field definitions to arguments collection
args.setFieldDefinition(field_defs)

# COMMAND ----------

# MAGIC %md Lastly, we need to configure a few settings that affect Zingg performance.  
# MAGIC
# MAGIC On Databricks, Zingg runs as a distributed process.  We want to ensure that Zingg can more fully take advantage of the distributed processing capabilities of the platform by dividing the data across some number of partitions aligned with the computational resources of our Databricks cluster. 
# MAGIC
# MAGIC We also want Zingg to sample our data at various stages.  Too big of a sample and Zingg will run slowly.  Too small a sample and Zingg will struggle to find enough samples to learn.  We typically will use a sample size between 0.0001 and 0.1, but finding the right value for a given dataset is more art that science:

# COMMAND ----------

# DBTITLE 1,Config Performance Settings
# define number of partitions to distribute data across
args.setNumPartitions( sc.defaultParallelism * 20 ) # default parallelism reflects databricks's cluster capacity

# define sample size
args.setLabelDataSampleSize(0.1)  

# COMMAND ----------

# MAGIC %md With all our Zingg configurations defined, we can now setup the Zingg client.  The client is configured for specific tasks.  The first task we will focus on is the [findTrainingData](https://docs.zingg.ai/zingg/stepbystep/createtrainingdata/findtrainingdata) task which tests various techniques for identifying matching data:

# COMMAND ----------

# DBTITLE 1,Define findTrainingData Task
# define task
findTrainingData_options = ClientOptions([ClientOptions.PHASE, 'findTrainingData'])

# configure findTrainingData task
findTrainingData = ZinggWithSpark(args, findTrainingData_options)

# initialize task
findTrainingData.init()

# COMMAND ----------

# MAGIC %md When we are done labeling the data generated through (multiple iterations of) the *findTrainingData* task, we will need to launch the *trainMatch* task.  This task combines two smaller tasks, *i.e.* *[train](https://docs.zingg.ai/zingg/stepbystep/train)* and *[match](https://docs.zingg.ai/zingg/stepbystep/match)*, which train the Zingg model using the labeled data and generate output containing potential matches from the initial (input) dataset:

# COMMAND ----------

# DBTITLE 1,Define trainMatch Task
# define task
trainMatch_options = ClientOptions([ClientOptions.PHASE, 'trainMatch'])

# configure findTrainingData task
trainMatch = ZinggWithSpark(args, trainMatch_options)

# initialize task
trainMatch.init()

# COMMAND ----------

# DBTITLE 1,Get Unmarked & Marked Folder Locations
# this is where Zingg stores unmarked candidate pairs produced by the findTrainData task
UNMARKED_DIR = findTrainingData.getArguments().getZinggTrainingDataUnmarkedDir() 

# this is where you store your marked candidate pairs that will be read by the Zingg train task
MARKED_DIR = findTrainingData.getArguments().getZinggTrainingDataMarkedDir() 

# COMMAND ----------

# MAGIC %md ##Step 2: Train the Model & Perform Initial Match
# MAGIC
# MAGIC With a set of labeled data in place, we can now *train* our Zingg model.  A common action performed immediately after this is the identification of duplicates (through a *match* operation) within the initial dataset.  While we could implement these in two tasks, these so frequently occur together that Zingg provides an option to combine the two tasks into one action:

# COMMAND ----------

# DBTITLE 1,Clear Old Outputs from Any Prior Runs
dbutils.fs.rm(output_dir, recurse=True)

# COMMAND ----------

# DBTITLE 1,Execute the trainMatch Step
trainMatch.execute()

# COMMAND ----------

# MAGIC %md The end result of the *train* portion of this task is a Zingg model, housed in the Zingg model directory.  If we ever want to retrain this model, we can re-run the findTrainingData step and add more labeled pairs to our marked dataset and re-run this action.  We may need to do that if we notice a lot of poor quality matches or if Zingg complains about having insufficient data to train our model.  But once this is run, we should verify a model has been recorded in the Zingg model directory:
# MAGIC
# MAGIC **NOTE** If Zingg complains about *insufficient data*, you may want to restart your Databricks cluster and retry the step before creating additional labeled pairs.

# COMMAND ----------

# DBTITLE 1,Examine Zingg Model Folder
display(
  dbutils.fs.ls( trainMatch.getArguments().getZinggModelDir() )
  )

# COMMAND ----------

# MAGIC %md The end result of the *match* portion of the last task are outputs representing potential duplicates in our dataset.  We can examine those outputs as follows:

# COMMAND ----------

# DBTITLE 1,Examine Initial Output
matches = (
  spark
    .read
    .format('delta')
    .option('path', output_dir)
    .load()
    .orderBy('z_cluster')
  )

display(matches)

# COMMAND ----------

# MAGIC %md In the *match* output, we see records associated with one another through a *z_cluster* assignment.  The strongest and weakest association of that record with any other member of the cluster is reflected in the *z_minScore* and *z_maxScore* values, respectively.  We will want to consider these values as we decide which matches to accept and which to reject.
# MAGIC
# MAGIC Also, we can look at the number of records within each cluster.  Most clusters will consist of 1 or 2 records.  Thta said there will be other clusters with abnormally large numbers of entries that desere a bit more scrutiny:

# COMMAND ----------

# DBTITLE 1,Examine Number of Clusters by Record Count
display(
  matches
    .groupBy('z_cluster')
      .agg(
        fn.count('*').alias('records')
        )
    .groupBy('records')
      .agg(
        fn.count('*').alias('clusters')
        )
    .orderBy('records')
  )

# COMMAND ----------

# MAGIC %md As we review the match output, it seems it would be helpful if Zingg provided some high-level metrics and diagnostics to help us understand the performance of our model.  The reality is that outside of evaluation scenarios where we may have some form of ground-truth against to evaluate our results, its very difficult to clearly identify the precision of a model such as this.  Quite often, the best we can do is review the results and make a judgement call based on the volume of identifiable errors and the patterns associated with those errors to develop a sense of whether our model's performance is adequate for our needs.
# MAGIC
# MAGIC Even if we have a high-performing model, we will have members we will have some matches that we will need to reject.  For this exercise, we will accept all the cluster assignment suggestions but in a real-world implementation, you'd want to accept only those cluster and cluster member assignments where all the entries are above a given threshold, *i.e.* *z_minScore*.  Any clusters not meeting this criteria would require a manual review.
# MAGIC
# MAGIC If we are satisfied with our results, we can capture our cluster data to a table structure that will allow us to more easily perform incremental data processing. Please note that we are assigning our own unique identifier to the clusters as the *z_cluster* value Zingg assigns is specific to the output associated with a task:

# COMMAND ----------

# DBTITLE 1,Get Clusters & IDs
# custom function to generate a guid string
@fn.udf('string')
def guid():
  return str(uuid.uuid1())

# get distinct clusters and assign a guid to each
clusters = (
  matches
    .select('z_cluster')
    .distinct()
    .withColumn('cluster_id', guid())
    .cache() # cache to avoid re-generating guids with subsequent calls
  )

clusters.count() # force full dataset into memory

display( clusters )

# COMMAND ----------

# MAGIC %md And now we can persist our data to tables named *clusters* and *cluster_members*:

# COMMAND ----------

# DBTITLE 1,Write Clusters
_ = (
  clusters
    .select('cluster_id')
    .withColumn('datetime', fn.current_timestamp())
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('clusters')
    )

display(spark.table('clusters'))

# COMMAND ----------

# DBTITLE 1,Write Cluster Members
_ = (
  matches
    .join(clusters, on='z_cluster')
    .selectExpr(
      'cluster_id',
      'recid',
      'givenname',
      'surname',
      'suburb',
      'postcode',
      'z_minScore',
      'z_maxScore',
      'current_timestamp() as datetime'
      )
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('cluster_members')
    )

display(spark.table('cluster_members'))

# COMMAND ----------

# MAGIC %md Upon review of the persisted data, it's very likely we will encounter records we need to correct.  Using standard [DELETE](https://docs.databricks.com/sql/language-manual/delta-delete-from.html), [UPDATE](https://docs.databricks.com/sql/language-manual/delta-update.html) and [INSERT](https://docs.databricks.com/sql/language-manual/sql-ref-syntax-dml-insert-into.html) statements, we can update the delta lake formatted tables in this environment to achieve the results we require.
# MAGIC
# MAGIC But what happens if we decide to retrain our model after we've setup these mappings? As mentioned above, retraining our model will cause new clusters with overlapping integer *z_cluster* identifiers to be created.  In this scenario, you need to decide whether you wish to preserve any manually adjusted mappings from before or otherwise start over from scratch.  If starting over, then simply drop and recreate the *clusters* and *cluster_members* tables. If preserving manually adjusted records, the GUID value associated with each cluster will keep the cluster identifiers unique.  You'll need to then decide how records assigned to clusters by the newly re-trained model should be merged with the preserved cluster data. It's a bit of a juggle so this isn't something you'll want to do on a regular basis.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | zingg                                  | entity resolution library | GNU Affero General Public License v3.0    | https://github.com/zinggAI/zingg/                       |
