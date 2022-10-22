# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/customer-er. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/customer-entity-resolution.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to setup the jobs needed by the remaining notebooks in the customer entity-resolution solution accelerator. 

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC In this notebook, we will be setting up the jobs that will be used to implement each of the two-phases of the Zingg workflow.  Each job is a [Spark Submit job](https://spark.apache.org/docs/latest/submitting-applications.html#launching-applications-with-spark-submit) referencing the Zingg JAR installed as described in the *ER Setup 00* notebook. Each will have access to its own [configuration file](https://docs.zingg.ai/zingg/stepbystep/configuration) based on specs provided by Zingg.

# COMMAND ----------

# DBTITLE 1,Get Config
# MAGIC %run "./00.0_ Intro & Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import requests, json, time
from copy import deepcopy

from pprint import PrettyPrinter
pp = PrettyPrinter()

# COMMAND ----------

# DBTITLE 1,Remove Any Prior Config Files
dbutils.fs.rm( config['dir']['config'], recurse=True)

# COMMAND ----------

# MAGIC %md ## Step 1: Verify Zingg JAR Installed
# MAGIC 
# MAGIC Before proceeding with job setup, be sure to have downloaded the Zingg JAR as described in *ER Setup 00*.  Here, we will quickly verify it's location:

# COMMAND ----------

# DBTITLE 1,Verify the Zingg JAR Installed
display(dbutils.fs.ls(config['job']['zingg jar path']))

# COMMAND ----------

# MAGIC %md ## Step 2: Build Config Files
# MAGIC 
# MAGIC Each Zingg job makes use of a config file. The first element of the configuration file that we will address will be the *data* element that defines how input files will be read and fields in those files interpreted.  To understand the attributes we will specify for this element, it helps to examine one of the input files: 

# COMMAND ----------

# DBTITLE 1,Examine Input Data File
file_name = dbutils.fs.ls(config['dir']['input'] + '/initial')[0].path

pp.pprint( dbutils.fs.head(file_name, 500) )

# COMMAND ----------

# MAGIC %md Our input files are a simple comma-separated value (CSV) file with fields of:</p>
# MAGIC 
# MAGIC * recid - the unique identifier for a voter on the voter registration system
# MAGIC * givenname - the given (first) name associated with a voter
# MAGIC * surname - the family (last) name associated with a voter
# MAGIC * suburb - the city within which the voter lives
# MAGIC * postcode - the postal code within which the voter lives
# MAGIC 
# MAGIC Leveraging this information, we might define our input files as follows:
# MAGIC 
# MAGIC **NOTE** The schema information associated with the input files is encoded as a multi-line string that will be evaluated to a dictionary by Zingg during job execution.

# COMMAND ----------

# DBTITLE 1,Define Inputs
data = [  # defined as a list as we may have multiple sets of input data
      {
        'name':'input',  # name you assign to the dataset
        'format':'csv',   # format of the dataset: csv or parquet
        'props': {        # various properties associated with the file
          'delimiter':',',  # comma delimiter
          'header':'true',  # has a header
          'location':config['dir']['input'] # path to folder holding files
           },
        'schema': # schema to apply to the data when read
            """{
            'type': 'struct',
            'fields':[
              {'name':'recid', 'type':'integer', 'nullable': true, 'metadata': {}},
              {'name':'givenname', 'type':'string', 'nullable': true, 'metadata': {}},
              {'name':'surname', 'type':'string', 'nullable': true, 'metadata': {}},
              {'name':'suburb', 'type':'string', 'nullable': true, 'metadata': {}},
              {'name':'postcode', 'type':'string', 'nullable': true, 'metadata': {}}
              ]
            }"""
        }
    ]

# COMMAND ----------

# MAGIC %md You may have noticed that we've defined the input folder path as the top-level input folder. This is something we will adjust below based on the needs of the specific job.  The same goes for the output folder location in the output configuration:

# COMMAND ----------

# DBTITLE 1,Define Outputs
output = [
  {
    'name':'output',  # name to assign to data outputs
    'format':'parquet',  # format to employ with data outputs: csv or parquet
    'props':{  # output file properties
      'location':config['dir']['output'],  # folder where to deposit outputs
      'overwrite':'true' # overwrite any existing data in output dir
      }
    }
  ]

# COMMAND ----------

# MAGIC %md Now that we have the basic configuration inputs and outputs defined, let's specify how incoming fields will be matched against other records and the output they will produce.  These input-output field mappings are captured in the *fieldDefinition* element.
# MAGIC 
# MAGIC Our incoming dataset has *recid*, *givenname*, *surname*, *suburb* and *postcode* fields.  As explained in the last notebook, the *recid* is a unique identifier left over from the original dataset from which this dataset was created.  Duplicate entries inserted by the dataset's authors can be identified through matching *recid* values (though there appear to be duplicates also in the original dataset). 
# MAGIC 
# MAGIC We won't use the *recid* ID to match records and instead will focus on voters' names and place of residence.  Supported match types are *FUZZY*, *EXACT*, *PINCODE*, *EMAIL*, *TEXT*, *NUMERIC*, and [many others](https://github.com/zinggAI/zingg/blob/031ed56945e8112c4c772b122cb0c40c67e59662/client/src/main/java/zingg/client/MatchType.java).  A match type of *DONT USE* will cause Zingg to ignore a field for matching purposes:
# MAGIC 
# MAGIC **NOTE** *fieldName* refers to the input field as defined in the *data* element and *fields* and *dataType* refer to the field to be produced as part of the defined *output*:

# COMMAND ----------

# DBTITLE 1,Define Field Mappings
fieldDefinition = [
    {'fieldName':'recid', 'matchType':'DONT USE', 'fields':'recid', 'dataType':'"integer"'},
    {'fieldName':'givenname', 'matchType':'FUZZY', 'fields':'first_name', 'dataType':'"string"'},
    {'fieldName':'surname', 'matchType':'FUZZY', 'fields':'last_name', 'dataType':'"string"'},
    {'fieldName':'suburb', 'matchType':'FUZZY', 'fields':'city', 'dataType':'"string"'},
    {'fieldName':'postcode', 'matchType':'FUZZY', 'fields':'zipcode', 'dataType':'"string"'}
    ]

# COMMAND ----------

# MAGIC %md And now we can put the configuration data together, incorporating various elements that will control the Zingg job.  These elements include the *labelDataSampleSize* which controls the random sampling rate for training the Zingg model, the *numPartitions* which controls the degree of distribution to apply to the Zingg job, and the *modelId* which assigns a name to the Zingg model:
# MAGIC 
# MAGIC **NOTE** We are defining the number of partitions to use based on the size of the current cluster.  If you adjust the cluster specs of your jobs, you may want to adjust the *numPartitions* configuration setting to align with your cluster size in order to maximize processing efficiency.

# COMMAND ----------

# DBTITLE 1,Assemble Base Config File
job_config = {
  'labelDataSampleSize':0.05,  # fraction of records to sample during model training exercises
  'numPartitions':config['job']['config']['num partitions'], # number of partitions against which to scale out data processing & model training
  'modelId': config['model name'],  # friendly name of model to be produced
  'zinggDir': config['dir']['zingg'], # folder within which to persist the model between steps
  'data': data,  # the input data
  'output': output, # the output data
  'fieldDefinition': fieldDefinition # the input-output field mappings
  }

# COMMAND ----------

# MAGIC %md The config file definition we've assembled can now be adjusted to address the slightly different needs of our initial and incremental data processing steps.  For the initial steps, *i.e.* the steps of identifying candidate pairs to label and then training a model, data output will be sent to an *initial* output folder.  For the incremental step, *i.e.* the step by which incoming data is linked to previously processed data, data output will be sent to an *incremental* output folder.  In addition, the incremental step will read both initial and incremental data files which have the same structure but which are found in different folder locations:

# COMMAND ----------

# DBTITLE 1,Define Config for Initial Steps 
# copy job config dictionary so that changes don't impact the base config
initial_job_config = deepcopy(job_config)

# adjust input settings
initial_job_config['data'][0]['name'] = initial_job_config['data'][0]['name'] + '_initial'
initial_job_config['data'][0]['props']['location'] = initial_job_config['data'][0]['props']['location'] + '/initial'

# adjust output settings
initial_job_config['output'][0]['name'] = initial_job_config['output'][0]['name'] + '_initial'
initial_job_config['output'][0]['props']['location'] = initial_job_config['output'][0]['props']['location'] + '/initial'

# display config
pp.pprint(initial_job_config)

# COMMAND ----------

# DBTITLE 1,Define Config for Incremental Match
# copy job config dictionary so that changes don't impact the base config
incremental_match_job_config = deepcopy(job_config)

# adjust input settings
incremental_match_job_config['data'][0]['name'] = incremental_match_job_config['data'][0]['name'] + '_incremental_match'
incremental_match_job_config['data'][0]['props']['location'] = incremental_match_job_config['data'][0]['props']['location'] + '/incremental/incoming'

# adjust output settings
incremental_match_job_config['output'][0]['name'] = incremental_match_job_config['output'][0]['name'] + '_incremental_match'
incremental_match_job_config['output'][0]['props']['location'] = incremental_match_job_config['output'][0]['props']['location'] + '/incremental/match'

# display config
pp.pprint(incremental_match_job_config)

# COMMAND ----------

# DBTITLE 1,Define Config for Incremental Link
# copy job config so that changes don't impact the base config
incremental_link_job_config = deepcopy(job_config)

# adjust input settings to support two inputs
incremental_link_job_config['data'] += deepcopy(incremental_link_job_config['data'])

# first input is priors
incremental_link_job_config['data'][0]['name'] = incremental_link_job_config['data'][0]['name'] + '_incremental_link_prior'
incremental_link_job_config['data'][0]['props']['location'] = incremental_link_job_config['data'][0]['props']['location'] + '/incremental/prior'

# second input is (new) incoming
incremental_link_job_config['data'][1]['name'] = incremental_link_job_config['data'][1]['name'] + '_incremental_link_incoming'
incremental_link_job_config['data'][1]['props']['location'] = incremental_link_job_config['data'][1]['props']['location'] + '/incremental/incoming'

# adjust output settings
incremental_link_job_config['output'][0]['name'] = incremental_link_job_config['output'][0]['name'] + '_incremental_link'
incremental_link_job_config['output'][0]['props']['location'] = incremental_link_job_config['output'][0]['props']['location'] + '/incremental/link'

# display incremental config
pp.pprint(incremental_link_job_config)

# COMMAND ----------

# MAGIC %md And now we can save our config data to actual file outputs:

# COMMAND ----------

# DBTITLE 1,Write Configs to Files
dbutils.fs.mkdirs(config['dir']['config'])

# define function to facilitate creation of json files
def write_config_file(config, local_file_path):
  with open(local_file_path, 'w') as fp:
    fp.write(json.dumps(config).replace("'", '\\"'))
    
# write config output
write_config_file(initial_job_config, '/dbfs'+ config['dir']['config'] + '/initial.json')
write_config_file(incremental_match_job_config, '/dbfs'+ config['dir']['config'] + '/incremental_match.json')
write_config_file(incremental_link_job_config, '/dbfs'+ config['dir']['config'] + '/incremental_link.json')

display(dbutils.fs.ls(config['dir']['config']))

# COMMAND ----------

# MAGIC %md ##Step 3: Setup Jobs
# MAGIC 
# MAGIC With the configuration files in place, we can now define the workflows (*jobs*) that will execute the Zingg logic. Each job will run on a dedicated jobs cluster and will be governed by a set of parameters defined as follows:
# MAGIC 
# MAGIC **NOTE** Some elements of the job specification are specific to the cloud environment on which you are running.  The easiest way to identify which elements are cloud-specific and which settings you may prefer to assign to each is to [manually create a temporary job](https://docs.databricks.com/data-engineering/jobs/jobs.html) and then review its JSON definition. 

# COMMAND ----------

# DBTITLE 1,Define Generic Job Specification
job_spec = {
  'new_cluster': {
      'spark_version': config['job']['config']['spark version'],
      'spark_conf': {
          'spark.databricks.delta.preview.enabled': 'true'
        },
      'node_type_id': config['job']['config']['node type id'],
      'spark_env_vars': {
          'PYSPARK_PYTHON': '/databricks/python3/bin/python3'
        },
      'enable_elastic_disk': 'true',
      'num_workers': int(config['job']['config']['num workers'])
    },
  'timeout_seconds': 0,
  'email_notifications': {},
  'name': 'find_training_job',
  'max_concurrent_runs': 1
  }

# COMMAND ----------

# MAGIC %md We can now complete the job spec definition for each of our three jobs. These jobs will run the Zingg library as a [Spark Submit job](https://spark.apache.org/docs/latest/submitting-applications.html#launching-applications-with-spark-submit). The parameters submitted with this job are captured as strings within a list as follows:

# COMMAND ----------

# DBTITLE 1,Initial Find Training Data Job Spec
initial_findTraining_jobspec = deepcopy(job_spec)

initial_findTraining_jobspec['name'] = config['job']['initial']['findTrainingData']

initial_findTraining_jobspec['spark_submit_task'] = {
      'parameters': [
          '--class', 'zingg.client.Client', # class within the zingg library (jar) to employ
          config['job']['zingg jar path'],         # path to zingg jar file
          '--phase=findTrainingData',       # job phase (to be employed by zingg)
          '--conf={0}'.format('/dbfs' + config['dir']['config'] + '/initial.json'), # local path to zingg conf file
          '--license=abc'                   # license to associate with zingg
          ]
      }

# COMMAND ----------

# DBTITLE 1,Initial Train Job Spec
initial_train_jobspec = deepcopy(job_spec)

initial_train_jobspec['name'] = config['job']['initial']['train']

initial_train_jobspec['spark_submit_task'] = {
      'parameters': [
          '--class',
          'zingg.client.Client',
          config['job']['zingg jar path'],
          '--phase=train',
          '--conf={0}'.format('/dbfs' + config['dir']['config'] + '/initial.json'),
          '--license=abc'
          ]
      }

# COMMAND ----------

# DBTITLE 1,Initial Match Job Spec
initial_match_jobspec = deepcopy(job_spec)

initial_match_jobspec['name'] = config['job']['initial']['match']

initial_match_jobspec['spark_submit_task'] = {
      'parameters': [
          '--class',
          'zingg.client.Client',
          config['job']['zingg jar path'],
          '--phase=match',
          '--conf={0}'.format('/dbfs' + config['dir']['config'] + '/initial.json'),
          '--license=abc'
          ]
      }

# COMMAND ----------

# DBTITLE 1,Incremental Link Job Spec
incremental_link_jobspec = deepcopy(job_spec)

incremental_link_jobspec['name'] = config['job']['incremental']['link']

incremental_link_jobspec['spark_submit_task'] = {
      'parameters': [
          '--class',
          'zingg.client.Client',
          config['job']['zingg jar path'],
          '--phase=link',
          '--conf={0}'.format('/dbfs' + config['dir']['config'] + '/incremental_link.json'),
          '--license=abc'
          ]
      }

# COMMAND ----------

# DBTITLE 1,Incremental Match Job Spec
incremental_match_jobspec = deepcopy(job_spec)

incremental_match_jobspec['name'] = config['job']['incremental']['match']

incremental_match_jobspec['spark_submit_task'] = {
      'parameters': [
          '--class',
          'zingg.client.Client',
          config['job']['zingg jar path'],
          '--phase=match',
          '--conf={0}'.format('/dbfs' + config['dir']['config'] + '/incremental_match.json'),
          '--license=abc'
          ]
      }

# COMMAND ----------

# MAGIC %md Now that we have the job specs defined, we can create the associated jobs as follows:

# COMMAND ----------

# DBTITLE 1,Create the Jobs
job_create_url = 'https://{0}/api/2.1/jobs/create'.format(config['job']['databricks workspace url'])
job_update_url = 'https://{0}/api/2.1/jobs/reset'.format(config['job']['databricks workspace url'])
headers = {"Authorization":"Bearer {0}".format(config['job']['api token'])}

for spec in [
  initial_findTraining_jobspec,
  initial_train_jobspec,
  initial_match_jobspec,
  incremental_link_jobspec,
  incremental_match_jobspec
  ]:
  
    # find job with this name
    try:
      job = ZinggJob(spec['name'], config['job']['databricks workspace url'], config['job']['api token'])
    except ValueError:
      pass
      job = None
      
    # create or update job:
    if job is None:
      # create new job
      resp = requests.post(job_create_url, headers=headers, json=spec)
    else:      
      # update the job with new settings
      resp = requests.post(job_update_url, headers=headers, json={'job_id':job.id, 'new_settings': spec})
    
    # get results of create/update
    try:
      resp.raise_for_status()
    except:
      print(resp.text)      

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
