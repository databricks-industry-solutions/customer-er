## Introduction

The process of matching records to one another is known as entity-resolution.  When dealing with entities such as persons, the process often requires the comparison of name and address information which is subject to inconsistencies and errors. In these scenarios, we often rely on probabilistic (*fuzzy*) matching techniques that identify likely matches based on degrees of similarity between these elements.

There are a wide range of techniques which can be employed to perform such matching.  The challenge is not just to identify which of these techniques provide the best matches but how to compare one record to all the other records that make up the dataset in an efficient manner.  Data Scientists specializing in entity-resolution often employ specialized *blocking* techniques that limit which customers should be compared to one another using mathematical short-cuts. 

In a [prior solution accelerator](https://databricks.com/blog/2021/05/24/machine-learning-based-item-matching-for-retailers-and-brands.html), we explored how some of these techniques may be employed (in a product matching scenario). In this solution accelerator, we'll take a look at how the [Zingg](https://github.com/zinggAI/zingg) library can be used to simplify an person matching implementation that takes advantage of a fuller range of techniques.

### The Zingg Workflow

Zingg is a library that provides the building blocks for ML-based entity-resolution using industry-recognized best practices.  It is not an application, but it provides the capabilities required to the assemble a robust application. When run in combination with Databricks, Zingg provides the application the scalability that's often needed to perform entity-resolution on enterprise-scaled datasets.

To build a Zingg-enabled application, it's easiest to think of Zingg as being deployed in two phases.  In the first phase, candidate pairs of potential duplicates are extracted from an initial dataset and labeled by expert users.  These labeled pairs are then used to train a model capable of scoring likely matches.

In the second phase, the model trained in the first phase is applied to newly arrived data.  Those data are compared to data processed in prior runs to identify likely matches between in incoming and previously processed dataset. The application engineer is responsible for how matched and unmatched data will be handled, but typically information about groups of matching records are updated with each incremental run to identify all the record variations believed to represent the same entity.
___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| zingg                                  | entity resolution library | GNU Affero General Public License v3.0    | https://github.com/zinggAI/zingg/                       |
| tabulate | pretty-print tabular data in Python | MIT License | https://pypi.org/project/tabulate/ |
| filesplit | Python module that is capable of splitting files and merging it back | MIT License | https://pypi.org/project/filesplit/ |

To run this accelerator, clone this repo into a Databricks workspace. Attach the RUNME notebook to any cluster running a DBR 11.0 or later runtime, and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. Execute the multi-step-job to see how the pipeline runs.

The job configuration is written in the RUNME notebook in json format. The cost associated with running the accelerator is the user's responsibility.

