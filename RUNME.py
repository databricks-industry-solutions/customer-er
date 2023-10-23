# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to illustrate the order of execution. Happy exploring! 
# MAGIC 🎉
# MAGIC
# MAGIC **Steps**
# MAGIC 1. Simply attach this notebook to a cluste and hit Run-All for this notebook. A multi-step job and the clusters used in the job will be created for you and hyperlinks are printed on the last block of the notebook. 
# MAGIC
# MAGIC 2. Run the accelerator notebooks: Feel free to explore the multi-step job page and **run the Workflow**, or **run the notebooks interactively** with the cluster to see how this solution accelerator executes. 
# MAGIC
# MAGIC     2a. **Run the Workflow**: Navigate to the Workflow link and hit the `Run Now` 💥. 
# MAGIC   
# MAGIC     2b. **Run the notebooks interactively**: Attach the notebook with the cluster(s) created and execute as described in the `job_json['tasks']` below.
# MAGIC
# MAGIC **Prerequisites** 
# MAGIC 1. You need to have cluster creation permissions in this workspace.
# MAGIC
# MAGIC 2. In case the environment has cluster-policies that interfere with automated deployment, you may need to manually create the cluster in accordance with the workspace cluster policy. The `job_json` definition below still provides valuable information about the configuration these series of notebooks should run with. 
# MAGIC
# MAGIC **Notes**
# MAGIC 1. The pipelines, workflows and clusters created in this script are not user-specific. Keep in mind that rerunning this script again after modification resets them for other users too.
# MAGIC
# MAGIC 2. If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators may require the user to set up additional cloud infra or secrets to manage credentials. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

# MAGIC %sh 
# MAGIC rm -rf /dbfs/tmp/solacc/customer_er/jar/
# MAGIC mkdir -p /dbfs/tmp/solacc/customer_er/jar/
# MAGIC cd /dbfs/tmp/solacc/customer_er/jar/
# MAGIC wget https://github.com/zinggAI/zingg/releases/download/v0.3.4/zingg-0.3.4-SNAPSHOT-spark-3.1.2.tar.gz
# MAGIC tar -xvf zingg-0.3.4-SNAPSHOT-spark-3.1.2.tar.gz 

# COMMAND ----------

job_json = {
        "timeout_seconds": 28800,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_automation",
            "group": "RCG"
        },
        "tasks": [
            {
                "job_cluster_key": "zingg_cluster",
                "libraries": [],
                "notebook_task": {
                    "notebook_path": f"00_Intro_&_Config"
                },
                "task_key": "00",
                "description": ""
            },
            {
                "job_cluster_key": "zingg_cluster",
                "notebook_task": {
                    "notebook_path": f"01_Prepare_Data"
                },
                "task_key": "01",
                "depends_on": [
                    {
                        "task_key": "00"
                    }
                ]
            },
            {
                "job_cluster_key": "zingg_cluster",
                "notebook_task": {
                    "notebook_path": f"02_Initial_Workflow_Part_A"
                },
                "task_key": "02",
                "depends_on": [
                    {
                        "task_key": "01"
                    }
                ],
                "libraries": [
                    {
                        "jar": "dbfs:/tmp/solacc/customer_er/jar/zingg-0.3.4-SNAPSHOT/zingg-0.3.4-SNAPSHOT.jar"
                    }
                ]
            },
            {
                "job_cluster_key": "zingg_cluster2",
                "notebook_task": {
                    "notebook_path": f"02_Initial_Workflow_Part_B"
                },
                "task_key": "03",
                "depends_on": [
                    {
                        "task_key": "02"
                    }
                ],
                "libraries": [
                    {
                        "jar": "dbfs:/tmp/solacc/customer_er/jar/zingg-0.3.4-SNAPSHOT/zingg-0.3.4-SNAPSHOT.jar"
                    }
                ]
            },
            {
                "job_cluster_key": "zingg_cluster",
                "notebook_task": {
                    "notebook_path": f"03_Incremental_Workflow"
                },
                "task_key": "04",
                "depends_on": [
                    {
                        "task_key": "03"
                    }
                ],
                "libraries": [
                    {
                        "jar": "dbfs:/tmp/solacc/customer_er/jar/zingg-0.3.4-SNAPSHOT/zingg-0.3.4-SNAPSHOT.jar"
                    }
                ]
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "zingg_cluster",
                "new_cluster": {
                    "spark_version": "12.2.x-scala2.12",
                "spark_conf": {
                    "spark.databricks.delta.formatCheck.enabled": "false"
                    },
                    "num_workers": 4,
                    "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_DS3_v2", "GCP": "n1-highmem-4"}, # different from standard API
                    "custom_tags": {
                        "usage": "solacc_automation"
                    },
                }
            },
            {
                "job_cluster_key": "zingg_cluster2",
                "new_cluster": {
                    "spark_version": "12.2.x-scala2.12",
                "spark_conf": {
                    "spark.databricks.delta.formatCheck.enabled": "false"
                    },
                    "num_workers": 4,
                    "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_DS3_v2", "GCP": "n1-highmem-4"}, # different from standard API
                    "custom_tags": {
                        "usage": "solacc_automation"
                    },
                }
            }
        ]
    }



# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
nsc = NotebookSolutionCompanion()
nsc.deploy_compute(job_json, run_job=run_job)
