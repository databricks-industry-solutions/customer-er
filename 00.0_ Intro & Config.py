# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/customer-er. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/customer-entity-resolution.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is provided an overview of the content in the other notebooks in this accelerator and to make accessible to these notebooks a set of standard configuration values. 

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC The process of matching records to one another is known as entity-resolution.  When dealing with entities such as persons, the process often requires the comparison of name and address information which is subject to inconsistencies and errors. In these scenarios, we often rely on probabilistic (*fuzzy*) matching techniques that identify likely matches based on degrees of similarity between these elements.
# MAGIC 
# MAGIC There are a wide range of techniques which can be employed to perform such matching.  The challenge is not just to identify which of these techniques provide the best matches but how to compare one record to all the other records that make up the dataset in an efficient manner.  Data Scientists specializing in entity-resolution often employ specialized *blocking* techniques that limit which customers should be compared to one another using mathematical short-cuts. 
# MAGIC 
# MAGIC In a [prior solution accelerator](https://databricks.com/blog/2021/05/24/machine-learning-based-item-matching-for-retailers-and-brands.html), we explored how some of these techniques may be employed (in a product matching scenario). In this solution accelerator, we'll take a look at how the [Zingg](https://github.com/zinggAI/zingg) library can be used to simplify an person matching implementation that takes advantage of a fuller range of techniques.

# COMMAND ----------

# MAGIC %md ### The Zingg Workflow
# MAGIC 
# MAGIC Zingg is a library that provides the building blocks for ML-based entity-resolution using industry-recognized best practices.  It is not an application, but it provides the capabilities required to the assemble a robust application. When run in combination with Databricks, Zingg provides the application the scalability that's often needed to perform entity-resolution on enterprise-scaled datasets.
# MAGIC 
# MAGIC To build a Zingg-enabled application, it's easiest to think of Zingg as being deployed in two phases.  In the first phase, candidate pairs of potential duplicates are extracted from an initial dataset and labeled by expert users.  These labeled pairs are then used to train a model capable of scoring likely matches.
# MAGIC 
# MAGIC In the second phase, the model trained in the first phase is applied to newly arrived data.  Those data are compared to data processed in prior runs to identify likely matches between in incoming and previously processed dataset. The application engineer is responsible for how matched and unmatched data will be handled, but typically information about groups of matching records are updated with each incremental run to identify all the record variations believed to represent the same entity.

# COMMAND ----------

# MAGIC %md ### Building a Solution
# MAGIC 
# MAGIC A typical entity-resolution application will provide a nice UI for end-user interactions with the data and an accessible database from which downstream applications can access deduplicated data. To handle the back-office processing supported by Zingg, Databricks jobs (*workflows*) representing various tasks performed in the Zingg-enabled workflows are implemented and triggered through [Databricks REST API](https://docs.databricks.com/dev-tools/api/latest/index.html) calls originating from the UI.  To keep our deployment simple, we'll implement the jobs that would be expected in a typical deployment but then leverage Databricks notebooks to provide the UI experience.  The UI experience provided by notebooks is less than ideal for the type of user who would typically be performing expert review in a customer identity resolution scenario but is sufficient to work out the steps in the workflow prior to a more robust UI implementation. 
# MAGIC 
# MAGIC The Zingg-aligned jobs we will setup in the Databricks environment make use of a JAR file which must be uploaded into the Databricks workspace as a *workspace library* prior to their execution.  To access this JAR and create the required workspace library, you can perform the following steps manually:</p>
# MAGIC 
# MAGIC 1. Navigate to the [Zingg GitHub repo](https://github.com/zinggAI/zingg)
# MAGIC 2. Click on *Releases* (located on the right-hand side of repository page)
# MAGIC 3. Locate the latest release for your version of Spark (which was *zingg-0.3.3-SNAPSHOT-spark-3.1.2* at the time this notebook was written)
# MAGIC 4. Download the compiled, gzipped *tar* file (found under the *Assets* heading) to your local machine
# MAGIC 5. Unzip the *tar.gz* file and retrieve the *jar* file
# MAGIC 6. Use the file to create a JAR-based library in your Databricks workspace following [these steps](https://docs.databricks.com/libraries/workspace-libraries.html)
# MAGIC 
# MAGIC We provided a script in `./config/setup` that automates these steps and sets up the jar file and source data for you. The script is executed as part of the next notebook (`00.1`).

# COMMAND ----------

# MAGIC %md ### The Solution Accelerator Assets
# MAGIC 
# MAGIC This accelerator is divided into two high-level parts. In the first part, we focus on the setup of the environment required by the application.  In the second part, we implement the Zingg-enabled application workflow. 
# MAGIC 
# MAGIC The notebooks that make up the first part are:</p>
# MAGIC 
# MAGIC * **00.0 Intro & Config** - used to provide access to a common set of configuration settings
# MAGIC * **00.1 Setup 01: Prepare Data** - used to setup the data that will be matched/linked in the application
# MAGIC * **00.2 Setup 02: Prepare Jobs** - used to setup the Zingg jobs
# MAGIC 
# MAGIC The notebooks that make up the second part are:</p>
# MAGIC 
# MAGIC * **01: Initial Workflow** - implements the process of identifying candidate pairs and assigning labels to them.  From the labeled pairs, a model is trained and database structures are initialized.
# MAGIC * **02: Incremental Workflow** - implements the process of incrementally updating the database based on newly arriving records.

# COMMAND ----------

# MAGIC %md ## Configuration
# MAGIC 
# MAGIC To enable consistent settings across the notebooks in this accelerator, we establish the following configuration settings:

# COMMAND ----------

# DBTITLE 1,Initialize Configuration Variable
if 'config' not in locals():
  config = {}

# COMMAND ----------

# DBTITLE 1,Initialize Database
# set database name
config['database name'] = 'ncvoters'

# create database to house mappings
_ = spark.sql('CREATE DATABASE IF NOT EXISTS {0}'.format(config['database name']))

# set database as default for queries
_ = spark.catalog.setCurrentDatabase(config['database name'] )

# COMMAND ----------

# DBTITLE 1,Set Zingg Model Name
config['model name'] = 'ncvoters'

# COMMAND ----------

# MAGIC %md We'll need to house quite a bit of data in specific locations to support different stages of our work.  We'll target a folder structure as follows:</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/er_dir_structure4.png' width=250>
# MAGIC 
# MAGIC **NOTE** We will create several additional subfolders under many of the directories in this folder structure.  This will be done within the various notebooks as needed. 

# COMMAND ----------

# DBTITLE 1,Initialize Folder Structure
# mount path where files are stored
mount_path = '/tmp/ncvoters'

config['dir'] = {}
config['dir']['config'] = f'{mount_path}/config'
config['dir']['downloads'] = f'{mount_path}/downloads' # original unzipped data files
config['dir']['input'] = f'{mount_path}/input'
config['dir']['output'] = f'{mount_path}/output'
config['dir']['staging'] = f'{mount_path}/staging' # staging area for incremental files
config['dir']['zingg'] = f'{mount_path}/zingg' # zingg models and temp data

# make sure directories exist
for dir in config['dir'].values():
  dbutils.fs.mkdirs(dir)

# COMMAND ----------

# MAGIC %md The Zingg jobs will be separated into those that support the initial workflow and those that support the incremental workflow. While some jobs such as *zingg_initial_match* and *zingg_incremental_match* are fundamentally the same, we separate the jobs in this manner to simplify this deployment and to illustrate how you might support differing job configurations with each phase:</p>
# MAGIC 
# MAGIC Initial
# MAGIC * zingg_initial_findTrainingData
# MAGIC * zingg_initial_train
# MAGIC * zingg_initial_match
# MAGIC 
# MAGIC Incremental
# MAGIC * zingg_incremental_link
# MAGIC * zingg_incremental_match
# MAGIC 
# MAGIC Please note that in order to trigger these jobs, calls to the REST API will need to supply a Personal Access Token with appropriate permissions.  You will need to enter that PAT in the configuration below. More information on creating Personal Access Tokens can be found [here](https://docs.databricks.com/dev-tools/api/latest/authentication.html).

# COMMAND ----------

# DBTITLE 1,Initialize Job Settings
# job names
config['job'] = {}
config['job']['initial'] = {}
config['job']['initial']['findTrainingData'] = 'zingg_initial_findTrainingData'
config['job']['initial']['train'] = 'zingg_initial_train'
config['job']['initial']['match'] = 'zingg_initial_match'
config['job']['incremental'] = {}
config['job']['incremental']['link'] = 'zingg_incremental_link'
config['job']['incremental']['match'] = 'zingg_incremental_match'

# parameters used to setup job configurations
config['job']['config']={}
config['job']['config']['spark version'] = sc.getConf().get('spark.databricks.clusterUsageTags.effectiveSparkVersion')
config['job']['config']['node type id'] = sc.getConf().get('spark.databricks.clusterUsageTags.clusterNodeType')
config['job']['config']['driver node type id'] = sc.getConf().get('spark.databricks.clusterUsageTags.clusterNodeType')
config['job']['config']['num workers'] = 8 # feel free to adjust the cluster size here
config['job']['config']['num partitions'] = sc.defaultParallelism * 10
    
config['job']['zingg jar path'] = "dbfs:/tmp/solacc/customer_er/jar/zingg-0.3.3-SNAPSHOT/zingg-0.3.3-SNAPSHOT.jar"

# settings to launch zingg jobs via rest api
config['job']['databricks workspace url'] = spark.sparkContext.getConf().get('spark.databricks.workspaceUrl')
config['job']['api token'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

# MAGIC %md In addition, we will define a ZinggJob class which will make executing the workflow jobs via the REST API easier:

# COMMAND ----------

# DBTITLE 1,Define ZinggJob Class
import requests
import json
import time

class ZinggJob:
   
  name = None
  id = None
  url = None
  _headers = None
  
  
  def __init__(self, name, databricks_workspace_url, api_token):  
    
    # attribute assignments
    self.name = name
    self.url = databricks_workspace_url
    self._headers = {'Authorization': f'Bearer {api_token}', 'User-Agent':'zinggai_zingg'}
    
    # get job id (based on job name)
    self.id = self._get_job_id()
    if self.id is None:
      self = None # invalidate self
      raise ValueError(f"A job with the name '{name}' was not found.  Please create the required jobs before attempting to proceed.")
    
  def _get_job_id(self):
    
    job_id = None
    
    # get list of jobs in databricks workspace
    job_resp = requests.get(f'https://{self.url}/api/2.0/jobs/list', headers=self._headers)
    
    # Handle edge case where no jobs are present in the workspace, otherwise attempting to iterate over job_resp will throw an error
    if len(job_resp.json()) == 0:
        return None
      
    # find job by name
    for job in job_resp.json().get('jobs'):
        if job.get('settings').get('name')==self.name:
            job_id = job.get('job_id') 
            break
    return job_id
  
  def run(self):
    post_body = {'job_id': self.id}
    run_resp = requests.post(f'https://{self.url}/api/2.0/jobs/run-now', json=post_body, headers=self._headers)
    run_id = run_resp.json().get('run_id')
    return run_id
  
  def wait_for_completion(self, run_id):
    
    # seconds to sleep between checks
    sleep_seconds = 30
    start_time = time.time()
    
    # loop indefinitely
    while True:
      
      # retrieve job info
      resp = requests.get(f'https://{self.url}/api/2.0/jobs/runs/get?run_id={run_id}', headers=self._headers)
      
      #calculate elapsed seconds
      elapsed_seconds = int(time.time()-start_time)
      
      # get job lfe cycle state
      life_cycle_state = resp.json().get('state').get('life_cycle_state')
      
      # if terminated, then get result state & break loop
      if life_cycle_state == 'TERMINATED':
          result_state = resp.json().get('state').get('result_state')
          break
          
      # else, report to user and sleep
      else:
          if elapsed_seconds > 0:
            print(f'Job in {life_cycle_state} state at { elapsed_seconds } seconds since launch.  Waiting {sleep_seconds} seconds before checking again.', end='\r')
          
          time.sleep(sleep_seconds)
    
    # return results
    print(f'Job completed in {result_state} state after { elapsed_seconds } seconds.  Please proceed with next steps to process the records identified by the job.')
    print('\n')         
         
    return result_state

  def run_and_wait(self):
    return self.wait_for_completion(self.run())
    

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

# COMMAND ----------


