# Databricks notebook source
# MAGIC %md The purpose of this notebook is to retrieve and label data as part of the the initial workflow in the Zingg Person Entity-Resolution solution accelerator.  This notebook is available on https://github.com/databricks-industry-solutions/customer-er.

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC
# MAGIC The purpose of this notebook is to train a Zingg model using an initial dataset within which we know there are some duplicate records.  With Zingg, this initial workload is executed through three sequential tasks (though the third task is optional).  These tasks are:</p>
# MAGIC
# MAGIC 1. **findTrainingData** - identify pairs of records that are candidates for being a match.  The user will label these pairs as a match, no match or uncertain.
# MAGIC 2. **train** - train a Zingg model using the labeled pairs resulting from user interactions on data coming from (multiple iterations of) the findTrainingData task.
# MAGIC 3. **match** - use the trained Zingg model to match duplicate records in the initial dataset.
# MAGIC </p>
# MAGIC
# MAGIC **NOTE** Because *train* and *match* are so frequently run together, Zingg provides a combined *trainMatch* task that we will employ to minimize the time to complete our work. We will continue to speak of these as two separate tasks but they will be executed as a combined task in this notebook.
# MAGIC
# MAGIC To enable this sequence of tasks, we will need to configure Zingg to understand where to read the initial data as its input.  We will also need to specify an output location, though only the *match* task will write data to this location. 
# MAGIC
# MAGIC While the *train* task does not write any explicit output, it does persist trained model assets to the *zingg* model folder.  The *findTrainingData* task will also send intermidiary data outputs to an *unmarked* folder under the *zingg* model folder.  (It will also expect us to place some data in a *marked* frolder location under that same *zingg* model folder.) Understanding where these assets reside can be helpful should you need to troubleshoot certain issues you might encounter during the *findTrainingData* or *train* tasks.
# MAGIC
# MAGIC Before jumping into these steps, we first need to verify the Zingg application JAR is properly installed on this cluster and then install the Zingg Python wrapper which provides a simple API for using the application:

# COMMAND ----------

# DBTITLE 1,Verify Zingg JAR Installed
# set default zingg path
zingg_jar_path = None

# for each jar in the jars folder
for j in dbutils.fs.ls('/FileStore/jars') + dbutils.fs.ls('/tmp/solacc/customer_er/jar/'): # either manually extracted and uploaded jar or use the RUNME Notebook automation
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
# MAGIC %run "./00_Intro & Config"

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

# MAGIC %md ##Step 2: Label Training Data
# MAGIC
# MAGIC With our tasks defined, we can now focus on our first task, *i.e.* *findTrainData*, and the labeling of the candidate pairs it produces.  Within this step, Zingg will read an initial set of input data and from it generate a set of record pairs that it believes may be duplicates.  As "expert data reviewers", we will review each pair and label it as either a *Match* or *No Match*.  (We may also label it as *Uncertain* if we cannot determine if the records are a match.)  
# MAGIC
# MAGIC In order to learn which techniques tend to lead to good matching, we will need to perform the labeling step numerous times.  You will notice that some runs generate better results than others.  This is Zingg testing out different approaches.  You will want to iterate through this step numerous times until you accumulate enough labeled pairs to produce good model results.  We suggest starting with 40 or more matches before attempting to train your model, but if you find after training that you aren't getting good results, you can always re-run this step to add more labeled matches to the original set of labeled records.
# MAGIC
# MAGIC That said, if you ever need to start over with a given Zingg model, you will want to either change the Zingg directory being used to persist these labeled pairs or delete the Zingg directory altogether. We have provided a function to assist with that.  Be sure to set the *reset* flag used by the function appropriately for your needs:

# COMMAND ----------

# DBTITLE 1,Get Unmarked & Marked Folder Locations
# this is where Zingg stores unmarked candidate pairs produced by the findTrainData task
UNMARKED_DIR = findTrainingData.getArguments().getZinggTrainingDataUnmarkedDir() 

# this is where you store your marked candidate pairs that will be read by the Zingg train task
MARKED_DIR = findTrainingData.getArguments().getZinggTrainingDataMarkedDir() 

# COMMAND ----------

# DBTITLE 1,Reset the Zingg Dir
def reset_zingg():
  # drop entire zingg dir (including matched and unmatched data)
  dbutils.fs.rm(findTrainingData.getArguments().getZinggDir(), recurse=True)
  # drop output data
  dbutils.fs.rm(output_dir, recurse=True)
  return

# determine if to reset the environment
reset = False

if reset:
  reset_zingg()

# COMMAND ----------

# MAGIC %md To assist with the reading of unmarked and marked pairs, we have defined a simple function.  It's called at the top of the label assignment logic (later) to produce the pairs that will be presented to the user.  If no data is found that requires labeling, it triggers the Zingg *findTrainingData* task to generate new candidate pairs.  That task can take a while to complete depending on the volume of data and performance-relevant characteristics assigned in the tasks's configuration (above):

# COMMAND ----------

# DBTITLE 1,Define Candidate Pair Retrieval Function
# retrieve candidate pairs
def get_candidate_pairs():
  
  # define internal function to restrict recursive calls
  def _get_candidate_pairs(depth=0):
  
    # initialize marked and unmarked dataframes to enable
    # comparisons (even when there is no data on either side)
    unmarked_pd = pd.DataFrame({'z_cluster':[]})
    marked_pd = pd.DataFrame({'z_cluster':[]})
  
    # read unmarked pairs
    try:
        tmp_pd = pd.read_parquet(
            '/dbfs'+ findTrainingData.getArguments().getZinggTrainingDataUnmarkedDir(), 
            engine='pyarrow'
        )
        if tmp_pd.shape[0] != 0: unmarked_pd = tmp_pd
    except:
        pass
  
    # read marked pairs
    try:
        tmp_pd = pd.read_parquet(
            '/dbfs'+ findTrainingData.getArguments().getZinggTrainingDataMarkedDir(),
            engine='pyarrow'
        )
        if tmp_pd.shape[0] != 0: marked_pd = tmp_pd
    except:
        pass
   
    # get unmarked not in marked
    candidate_pairs_pd = unmarked_pd[~unmarked_pd['z_cluster'].isin(marked_pd['z_cluster'])]
    candidate_pairs_pd.reset_index(drop=True, inplace=True)
  
    # test to see if anything found to label:
    if depth > 1: # too deep, leave
      return candidate_pairs_pd
    
    elif candidate_pairs_pd.shape[0] == 0: # nothing found, trigger zingg and try again
   
      print('No unmarked candidate pairs found.  Running findTraining job ...','\n')
      findTrainingData.execute()
      
      candidate_pairs_pd = _get_candidate_pairs(depth+1)
    
    return candidate_pairs_pd
  
  
  return _get_candidate_pairs()

# COMMAND ----------

# MAGIC %md Now we can present our candidate pairs for labeling.  We are using [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/) to make the presentation of these data and the assignment of labels a bit more presentable, but a Databricks notebook is not intended to replace a proper end user UI.
# MAGIC
# MAGIC To assign labels to a pair, run the cell below.  Once the data are presented, you can use the provided buttons to mark each pair.  When you are done, you can save your label assignments by running the cell that immediately follows.  Once you have accumulated a sufficient number of matches - 40 should be used as a minimum for most datasets - you can move on to subsequent steps.  Until you have accumulated the required amount, you will need to repeatedly run these cells (remembering to save following label assignment) until you've hit your goal:

# COMMAND ----------

# DBTITLE 1,Label Training Set
# define variable to avoid duplicate saves
ready_for_save = False

# user-friendly labels and corresponding zingg numerical value
# (the order in the dictionary affects how displayed below)
LABELS = {
  'Uncertain':2,
  'Match':1,
  'No Match':0  
  }

# GET CANDIDATE PAIRS
# ========================================================
candidate_pairs_pd = get_candidate_pairs()
n_pairs = int(candidate_pairs_pd.shape[0]/2)
# ========================================================

# DEFINE IPYWIDGET DISPLAY
# ========================================================
display_pd = candidate_pairs_pd.drop(labels=['z_zid', 'z_prediction', 'z_score', 'z_isMatch'], axis=1)

# define header to be used with each displayed pair
html_prefix = "<p><span style='font-family:Courier New,Courier,monospace'>"
html_suffix = "</p></span>"
header = widgets.HTML(value=f"{html_prefix}<b>" + "<br />".join([str(i)+"&nbsp;&nbsp;" for i in display_pd.columns.to_list()]) + f"</b>{html_suffix}")

# initialize display
vContainers = []
vContainers.append(widgets.HTML(value=f'<h2>Indicate if each of the {n_pairs} record pairs is a match or not</h2>'))

# for each set of pairs
for n in range(n_pairs):

  # get candidate records
  candidate_left = display_pd.loc[2*n].to_list()
  candidate_right = display_pd.loc[(2*n)+1].to_list()

  # reformat candidate records for html
  left = widgets.HTML(value=html_prefix + "<br />".join([str(i) for i in candidate_left]) + html_suffix)
  right = widgets.HTML(value=html_prefix + "<br />".join([str(i) for i in candidate_right]) + html_suffix)

  # define pair for presentation
  presented_pair = widgets.HBox(children=[header, left, right])

  # assign label options to pair
  label = widgets.ToggleButtons(
    options=LABELS.keys(), 
    button_style='info'
    )

  # define blank line between displayed pair and next
  blankLine=widgets.HTML(value='<b>' + '-'.rjust(105,"-") + '</b>')

  # append pair, label and blank line to widget structure
  vContainers.append(widgets.VBox(children=[presented_pair, label, blankLine]))

# present widget
display(widgets.VBox(children=vContainers))
# ========================================================

# mark flag to allow save 
ready_for_save = True

# COMMAND ----------

# DBTITLE 1,Save Assigned Labels
if not ready_for_save:
  print('No labels have been assigned. Run the previous cell to create candidate pairs and assign labels to them before re-running this cell.')

else:

  # ASSIGN LABEL VALUE TO CANDIDATE PAIRS IN DATAFRAME
  # ========================================================
  # for each pair in displayed widget
  for pair in vContainers[1:]:

    # get pair and assigned label
    html_content = pair.children[0].children[1].get_interact_value() # the displayed pair as html
    user_assigned_label = pair.children[1].get_interact_value() # the assigned label

    # extract candidate pair id from html pair content
    str_beg = len(html_prefix)
    str_end = html_content.index("<br />")
    pair_id = html_content[str_beg:str_end] # aka z_cluster

    # assign label to candidate pair entry in dataframe
    candidate_pairs_pd.loc[candidate_pairs_pd['z_cluster']==pair_id, 'z_isMatch'] = LABELS.get(user_assigned_label)
  # ========================================================

  # SAVE LABELED DATA TO ZINGG FOLDER
  # ========================================================
  # make target directory if needed
  dbutils.fs.mkdirs(MARKED_DIR)

  # save label assignments
  candidate_pairs_pd.to_parquet(
    '/dbfs' + MARKED_DIR + f'/markedRecords_'+ str(time.time_ns()/1000) + '.parquet', 
    compression='snappy',
    index=False, # do not include index
    engine='pyarrow'
    )
  # ========================================================

  # COUNT MARKED MATCHES 
  # ========================================================
  marked_matches = 0
  try:
    tmp_pd = pd.read_parquet( '/dbfs' + MARKED_DIR, engine='pyarrow')
    marked_matches = int(tmp_pd[tmp_pd['z_isMatch'] == LABELS['Match']].shape[0] / 2)
  except:
    pass

  # show current status of process
  print('Labels saved','\n')
  print(f'You now have labeled {marked_matches} matches.')
  print("If you need more pairs to label, re-run the previous cell and assign more labels.")
  # ========================================================  

  # save completed
  ready_for_save = False

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Before moving on to the next phase, it's a good idea to review the labels assigned to the candidate pairs for errors:

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
# MAGIC user_assigned_label = 'Match | Unmatch | Uncertain'
# MAGIC
# MAGIC
# MAGIC
# MAGIC # read existing data
# MAGIC marked_pd = pd.read_parquet(
# MAGIC       '/dbfs'+ MARKED_DIR, 
# MAGIC       engine='pyarrow'
# MAGIC        )
# MAGIC
# MAGIC # assign new label
# MAGIC marked_pd.loc[ marked_pd['z_cluster']==z_cluster, 'z_isMatch'] = LABELS.get(user_assigned_label)
# MAGIC
# MAGIC # delete old records
# MAGIC dbutils.fs.rm(MARKED_DIR, recurse=True)
# MAGIC dbutils.fs.mkdirs(MARKED_DIR)
# MAGIC
# MAGIC # write updated records
# MAGIC marked_pd.to_parquet(
# MAGIC    '/dbfs' + MARKED_DIR +'/markedRecords_'+ str(time.time_ns()/1000) + '.parquet', 
# MAGIC     compression='snappy',
# MAGIC     index=False, # do not include index
# MAGIC     engine='pyarrow'
# MAGIC     )
# MAGIC
# MAGIC ```

# COMMAND ----------

# MAGIC %md ##Step 3: Train the Model & Perform Initial Match
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
