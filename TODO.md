# Following tasks are to be done:

## Setup
- [X] Access AWS client services
- [X] Access bucket contents
- [X] Uploading and downloading files

## Data pipeline
- [X] Fetching the latest code for data preparation - Container gets automatically updated
- [X] Automatic data preparation whenever new data is uploaded - Triggered via Lambda function
- [X] Upload prepped tensor datasets to S3 - Text data is being uploaded
- [ ] Check if Lambda function image URI can be automatically updated when the ECR image is updated
- [ ] Run Lambda using its URL when the ECR image is run
- [ ] Split data into train and test

## Training
- [X] Training pipeline execution with Optuna hyperparameter tuning
- [X] Consistent data generation in data module
- [X] Deterministic run of each trial for ensuring reproducibility - this prevents saving a lot of models and spiking up the S3 cost
- [X] Downloading data into training module directly from S3
- [ ] Experiment results tracking and model results visualization
- [ ] Saving experiment results to S3
- [ ] Generating different Optuna hyperparameter runs as experiment runs?

## Post training activities: Deployment and further
- [ ] Performance review among models
- [ ] Best model selection and approval
- [ ] Transfer of model to higher environments
- [ ] Model fairness evaluation

# Pipelines:

The pipelines are decoupled. This decoupling controls changes in one pipeline from affecting other pipelines.

## Data

`Trigger: File upload in S3 bucket in raw/` -> Detected by `AWS Lambda` 

* Fetches new data from different sources and keeps raw data in `S3`
* There is some data preparation code which runs. This code is executed via `AWS Lambda`
* The data transformation code is stored in a container in `ECR`. This image is automatically updated via `Github Actions` whenever there is code change in the data_pipeline folder.
* After data processing the data is written out again in `S3` to be consumed in training jobs
* Pickles out data transformer class object to `S3`

## Training

`Trigger: File upload in S3 bucket in processing/`  -  Not implemented currently.

* Training is done through containers in Sagemaker using Optuna?
* Experiments are tracked in Sagemaker

## Deployment

`Trigger:`  -   Not implemented currently.

* Model selection, approval and registration
* Endpoint creation
* Model deployment to endpoint

## Monitoring

`Trigger: Biweekly cron jobs`

* Data and concept drift implementation

# Additional ideas

* Pipeline monitoring
* Pipelines' testing
* Merge rules, UAT, regression testing, SIT and code coverage
* Event based triggers
* Underlying infrastructure monitoring
* Log analytics

