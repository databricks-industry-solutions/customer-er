# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/customer-er. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/customer-entity-resolution.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to setup the data assets needed by the remaining notebooks in the customer entity-resolution solution accelerator. 

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC For this solution accelerator, we'll make use of the [North Carolina Voters 5M](https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution) dataset made available by the [Database Group Leipzig](https://dbs.uni-leipzig.de/en). This dataset, more fully documented in [this paper](https://dbs.uni-leipzig.de/file/famer-adbis2017.pdf), contains name and limited address information for several million registered voters within the state of North Carolina. There are purposefully duplicate records inserted in the set with specific adjustments to make them fuzzy matchable bringing the total number of records in the dataset to around 5-million, hence the name of the dataset.
# MAGIC 
# MAGIC The dataset is made available for download as a gzipped TAR file which needs to be downloaded, unzipped, untarred and uploaded to a folder named *downloads* under a [mount point](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs) in your environment before running the remainder of these notebooks. In our environment, we've used a default name of */mnt/ncvoters* for our mount point.  You can alter this in the *ER Setup 00* notebook if you've elected to create a mount point under a different name.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install filesplit

# COMMAND ----------

# DBTITLE 1,Get Config
# MAGIC %run "./00.0_ Intro & Config"

# COMMAND ----------

# DBTITLE 1,Download data and Zingg jar
# MAGIC %run "./config/setup"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as f

from filesplit.split import Split

# COMMAND ----------

# MAGIC %md ## Step 1: Reset the Environment
# MAGIC 
# MAGIC Zingg depends on having just the right data in the right place.  To ensure we maintain a clean environment, we'll reset all the directories housing input, output and transient data.  In most environments, such a step should not be necessary.  This is just a precaution:

# COMMAND ----------

# DBTITLE 1,Reset the Data Environment
dbutils.fs.rm(config['dir']['input'], recurse=True)
dbutils.fs.rm(config['dir']['output'], recurse=True)
dbutils.fs.rm(config['dir']['staging'], recurse=True)

# COMMAND ----------

# MAGIC %md ## Step 2: Separate Data into Initial and Incremental Sets
# MAGIC 
# MAGIC The North Carolina Voters 5M dataset consists of 5 files containing roughly 1-million records each. We will use the data in the first 4 files as our initial dataset and then split the 5th file into incremental sets of approximately 10,000 records each:

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
    
  # if 4: split into files with n customers at time copy to incremental folder
  elif file_num == 4:
    
    dbutils.fs.mkdirs(config['dir']['staging'])
    
    # split file into files of 10,000 records each
    split = Split('/'+ file.path.replace(':',''), '/dbfs' + config['dir']['staging'])
    split.bylinecount(10000, includeheader=True)
    
    # cleanup manifest file created by Split
    dbutils.fs.rm(config['dir']['staging']+'/manifest')

# COMMAND ----------

# DBTITLE 1,Verify Initial Dataset Has 4 Files
display(
  dbutils.fs.ls(config['dir']['input'] + '/initial')
  )

# COMMAND ----------

# DBTITLE 1,Verify Staging Area for Incremental Data Is Populated
display(
  dbutils.fs.ls(config['dir']['staging'])
  )

# COMMAND ----------

# MAGIC %md ## Step 2: Examine the Data
# MAGIC 
# MAGIC To get a sense of the data, we'll examine the records in the original dataset:

# COMMAND ----------

# DBTITLE 1,Access Initial Data Set
data = (
  spark
    .read
    .csv(
      path=config['dir']['downloads'],
      sep=',',
      header=True,
      inferSchema=True
      )
    .createOrReplaceTempView('ncvoters')
  )

display(spark.table('ncvoters'))

# COMMAND ----------

# MAGIC %md In the dataset, voters are identified based on the following fields:</p>
# MAGIC 
# MAGIC * givenname
# MAGIC * surname
# MAGIC * suburb
# MAGIC * postcode
# MAGIC 
# MAGIC A unique identifier, *recid*, was used in the original dataset to identify unique records.
# MAGIC 
# MAGIC To create duplicates in the dataset, the team responsible for creating it simply re-inserted some number of the rows back into it without any modifications.  Another set of duplicates was created by re-inserting rows while *corrupting* one or multiple of the 4 fields identified above.  Corruptions take the form of the removal, replacement or reversal of some number or characters from within a string as would be typical of a poor data entry process.  These duplicates are identifiable by their duplicate *recid* values:

# COMMAND ----------

# DBTITLE 1,Identify Author-Generated Duplicates
# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM ncvoters
# MAGIC WHERE recid IN (
# MAGIC   SELECT recid -- duplicate recid values
# MAGIC   FROM ncvoters a
# MAGIC   GROUP BY recid
# MAGIC   HAVING COUNT(*) > 1
# MAGIC   )
# MAGIC ORDER BY recid

# COMMAND ----------

# MAGIC %md Still other duplicates are naturally occuring in the dataset.  With a dataset of this size, it's not unexpected that some errors were not caught following data entry.  For example, consider these records which appear to be exact duplicates but which have separate *recid* values.
# MAGIC 
# MAGIC It is possible that two individuals within a given zip code have the same first and last name so that some of these records only appear to be duplicates given the lack of additional identifying data in the dataset. However, the uniqueness of some of these names would indicate that some are true duplicates in the original dataset:

# COMMAND ----------

# DBTITLE 1,Identify Apparent Duplicates In Original Dataset
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   a.*
# MAGIC FROM ncvoters a
# MAGIC INNER JOIN ncvoters b
# MAGIC   ON a.givenname=b.givenname AND a.surname=b.surname AND a.suburb=b.suburb AND a.postcode=b.postcode
# MAGIC WHERE a.recid != b.recid
# MAGIC UNION
# MAGIC SELECT
# MAGIC   b.*
# MAGIC FROM ncvoters a
# MAGIC INNER JOIN ncvoters b
# MAGIC   ON a.givenname=b.givenname AND a.surname=b.surname AND a.suburb=b.suburb AND a.postcode=b.postcode
# MAGIC WHERE a.recid != b.recid
# MAGIC ORDER BY givenname, surname, suburb, postcode

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
