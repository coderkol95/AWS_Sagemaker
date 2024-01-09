# AWS MLOps template

MLOps implementation on AWS for a product team. Implements the following scenario:

Data arrives at a frequency or ad-hoc. Data transformation scripts are adjusted and updated automatically via docker image on Lambda. Data is transformed and kept in S3 on arrival via Lambda. Model traning is triggered when new data is available by updating few parameters. Best model hyperparameters are automatically selected using Optuna along with all model training runs. This model can then be deployed as required.

# Highlights

* No need to step out of VS code environment for data engineers and data scientists
* Guides on required IAM roles, users and other administrative tasks
* Automated model hyperparameter tuning to get best hyperparameters

# Data pipeline

* Data is picked up from a S3 location whenever it arrives
* Data transformation is done via Docker image in Lambda
* Any change in these scripts trigger new image upload to ECR and subsequently to Lambda
* Transformed data with data transformation objects are kept in S3

# Model pipeline

* Different modular model architectures can be created via docker images
* Parallely run different architectures with Optuna to compare results from different architectures
* Best model hyperparameters saved as a separate file
* No need to save model as runs are deterministic and results are reproducible - saves S3 costs
* Model run results stored in S3 for analysis in Tensorboard locally
