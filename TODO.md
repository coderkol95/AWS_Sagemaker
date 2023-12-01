# Following tasks are to be done:

- [X] Access AWS client services
- [X] Access a bucket contents
- [X] Uploading and downloading files
- [X] Fetching the latest code for data preparation - Container gets automatically updated
- [X] Automatic data preparation whenever new data is uploaded - Triggered via Lambda function
- [X] Upload prepped tensor datasets to S3 - Text data is being uploaded
- [ ] Training pipeline execution with different hyperparameter selection
- [ ] Experiment results tracking and model results visualization
- [ ] Best model selection and approval
- [ ] Transfer of model to higher environments
- [ ] Check if Lambda function image URI can be automatically updated when the ECR image is updated

# Pipelines:

The pipelines are decoupled. This decoupling controls changes in one pipeline from affecting other pipelines.

## Data

`Trigger: File upload in S3 bucket in raw/` 

* Fetches new data from different sources and keeps raw data in `S3`
* There is some data preparation code which runs. This code is executed via `AWS Lambda`
* The data transformation code is stored in a container in `ECR`. This image is automatically updated via `Github Actions` whenever there is code change in the data_pipeline folder.
* After data processing the data is written out again in `S3` to be consumed in training jobs
* Pickles out data transformer class object to `S3`

## Training

`Trigger: File upload in S3 bucket in processing/`  -  Not implemented this trigger now.

* Training is done through containers in Sagemaker using Optuna?
* Experiments are tracked in Sagemaker

## Deployment

`Trigger: `

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

