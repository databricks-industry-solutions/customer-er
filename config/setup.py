# Databricks notebook source
# MAGIC %fs 
# MAGIC rm -r /tmp/solacc/customer_er/jar/

# COMMAND ----------

# MAGIC %fs 
# MAGIC mkdirs /tmp/solacc/customer_er/jar/

# COMMAND ----------

# MAGIC %fs 
# MAGIC rm -r /tmp/ncvoters/downloads/

# COMMAND ----------

# MAGIC %fs mkdirs /tmp/ncvoters/downloads/

# COMMAND ----------

# MAGIC %sh 
# MAGIC cd /dbfs/tmp/solacc/customer_er/jar/
# MAGIC wget https://github.com/zinggAI/zingg/releases/download/v0.3.3/zingg-0.3.3-SNAPSHOT-spark-3.1.2.tar.gz
# MAGIC tar -xvf zingg-0.3.3-SNAPSHOT-spark-3.1.2.tar.gz

# COMMAND ----------

# MAGIC %sh -e
# MAGIC rm -r /tmp/downloads
# MAGIC mkdir /tmp/downloads
# MAGIC cd /tmp/downloads
# MAGIC wget https://www.informatik.uni-leipzig.de/~saeedi/5Party-ocp20.tar.gz
# MAGIC tar -xvf 5Party-ocp20.tar.gz
# MAGIC rm -r /dbfs/tmp/ncvoters/downloads/
# MAGIC mkdir /dbfs/tmp/ncvoters/downloads/
# MAGIC cp -a /tmp/downloads/5Party-ocp20/. /dbfs/tmp/ncvoters/downloads/
# MAGIC ls /dbfs/tmp/ncvoters/downloads/

# COMMAND ----------


