# Databricks notebook source
# MAGIC %md The purpose of this notebook is provided an overview of the Zingg Person Entity-Resolution solution accelerator and to set the configuration values to be used throughout the remaining notebooks. This notebook is available on https://github.com/databricks-industry-solutions/customer-er.

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC
# MAGIC The process of matching records related to key business concepts (such as customers, products, *etc.*) to one another is known as entity-resolution.  When dealing with entities such as persons, the process often requires the comparison of name and address information which is subject to inconsistencies and errors. In this scenario, we often rely on probabilistic (*fuzzy*) matching techniques that identify likely matches based on degrees of similarity between these attributes.
# MAGIC
# MAGIC There are a wide range of techniques which can be employed to perform such matching.  The challenge is not just to identify which of these techniques provide the best matches but how to compare one record to all the other records that make up the dataset in an efficient manner.  Data Scientists specializing in entity-resolution often employ *blocking* techniques that limit which records are similar enough to one another to warrent a more detailed evaluation. As a result entity-resolution problems require familiarity with a breadth of techniques coupled with some domain knowledge, making this a challenge space for most non-specialists to enter.
# MAGIC
# MAGIC Fortunately, the [Zingg](https://www.zingg.ai/) library encapsulates these techniques, allowing Data Scientists and Engineers with limited experience to quickly make use of industry-recognized best practices and techniques for entity-resolution.  When run on Databricks, Zingg can tap into the scalabilty of the platform to make relatively quick work of large data matching workloads.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md ###Understanding Zingg
# MAGIC
# MAGIC To build a Zingg-enabled application, it's easiest to think of Zingg as being deployed in two phases.  In the first phase that we will refer to as the initial workflow, candidate pairs of potential duplicates are extracted from an initial dataset and labeled by expert users.  These labeled pairs are then used to train a model capable of scoring likely matches.
# MAGIC
# MAGIC In the second phase that we will refer to as the incremental workflow, the trained model is applied to newly arrived data.  Those data are compared to data processed in prior runs to identify likely matches between in incoming and previously processed dataset. The application engineer is responsible for how matched and unmatched data will be handled, but typically information about groups of matching records are updated with each incremental run to identify all the record variations believed to represent the same entity.
# MAGIC
# MAGIC The initial workflow must be run before we can proceed with the incremental workflow.  The incremental workflow is run whenever new data arrive that require linking and deduplication. If we feel that the model could be improved through additional training, we can perform additional cycles of record labeling by rerunning the initial workflow.  The retrained model will then be picked up in the next incremental run.</p>
# MAGIC
# MAGIC A typical entity-resolution application will provide a nice UI for end-user interactions with the data and an accesible database from which downstream applications can access deduplicated data. Our focus here is on the backend processes triggered by those user interactions.  We will use [ipywidgets](https://ipywidgets.readthedocs.io/) to enable some limited UI-capabilities, but in a real-world deployment, you should work with an application developer to provide user experience better aligned with the needs of a business-aligned user.

# COMMAND ----------

# MAGIC %md ### Installing Zingg
# MAGIC
# MAGIC To leverage Zingg, we'll need to first install the Zingg JAR file as a *workspace library* on our cluster.  To do this, please follow these steps:
# MAGIC </p>
# MAGIC
# MAGIC 1. Navigate to the releases associated with the [Zingg GitHub repo](https://github.com/zinggAI/zingg/releases)
# MAGIC 2. Click on *Releases* (located on the right-hand side of repository page)
# MAGIC 3. Locate the latest release for your version of Spark (which was *zingg-0.3.4-SNAPSHOT-spark-3.1.2* at the time this notebook was written)
# MAGIC 4. Download the compiled, gzipped *tar* file (found under the *Assets* heading) to your local machine
# MAGIC 5. Unzip the *tar.gz* file and retrieve the *jar* file
# MAGIC 6. Use the file to create a JAR-based library in your Databricks workspace following [these steps](https://docs.databricks.com/libraries/workspace-libraries.html)
# MAGIC
# MAGIC Alternatively you can run the **./RUNME** notebook and use the Workflow and Cluster created in that notebook to run this accelerator. The **RUNME** notebook automated the download, extraction and installation of the Zingg jar.
# MAGIC
# MAGIC At the top of each notebook in this accelerator, we will verify the JAR file has been properly installed just to make sure you don't encounter unexplained errors later in the code.  You will also see where we perform a *pip install* of the Zingg Python library.  This serves as a wrapper for the JAR which makes triggering Zingg jobs easier in Python.

# COMMAND ----------

# MAGIC %md ### Solution Accelerator Assets
# MAGIC
# MAGIC This accelerator is divided into a series of notebooks.  The role of these in the solution accelerator is as follows</p>
# MAGIC
# MAGIC * **00 Intro & Config** - implements configuration values used throughout the other notebooks and provides an overview of the solution accelerator.
# MAGIC * **01 Data Prep** - setups the data assets used in the solution accelerator.
# MAGIC * **02 Initial Workflow** - implements the process of identifying candidate pairs and assigning labels to them.  From the labeled pairs, a model is trained and database structures are initialized.
# MAGIC * **03 Incremental Workflow** - implements the process of incrementally updating the database based on newly arriving records.

# COMMAND ----------

# MAGIC %md ## Configuration
# MAGIC
# MAGIC To enable consistent settings across the notebooks in this accelerator, we establish the following configuration settings:

# COMMAND ----------

# DBTITLE 1,Initialize Configuration Variable
if 'config' not in locals():
  config = {}

# COMMAND ----------

# DBTITLE 1,Database
# set database name
config['database name'] = 'zingg_ncvoters'

# create database to house mappings
_ = spark.sql('CREATE DATABASE IF NOT EXISTS {0}'.format(config['database name']))

# set database as default for queries
_ = spark.catalog.setCurrentDatabase(config['database name'] )

# COMMAND ----------

# DBTITLE 1,Zingg Model
config['model name'] = 'zingg_ncvoters'

# COMMAND ----------

# MAGIC %md The Zingg workflow depends on access to various folder locations where the trained model and intermediary assets can be placed between various steps.  The purpose of these locations will be explained in the subsequent notebooks:

# COMMAND ----------

# DBTITLE 1,Directories
# path where files are stored
#mount_path = '/tmp/zingg_ncvoters'
mount_path = '/home/bryan.smithdatabricks.com/zingg_ncvoters'

config['dir'] = {}

# folder locations where you place your data
config['dir']['downloads'] = f'{mount_path}/downloads' # original unzipped data files that you will upload to the environment
config['dir']['input'] = f'{mount_path}/inputs' # folder where downloaded files will be seperated into initial and incremental data assets
config['dir']['tables'] = f'{mount_path}/tables' # location where external tables based on the data files will house data 

# folder locations Zingg writes data
config['dir']['zingg'] = f'{mount_path}/zingg' # zingg models and temp data
config['dir']['output'] = f'{mount_path}/output'

# make sure directories exist
for dir in config['dir'].values():
  dbutils.fs.mkdirs(dir)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | zingg                                  | entity resolution library | GNU Affero General Public License v3.0    | https://github.com/zinggAI/zingg/                       |
