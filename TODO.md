# Following tasks are to be done:

- [X] Access AWS client services
- [X] Access a bucket contents
- [X] Uploading and downloading files
- [X] Fetching the latest files for data preparation
- [ ] Automatic data preparation whenever new data is uploaded
- [ ] Upload prepped tensor datasets to S3
- [ ] Training pipeline execution with different hyperparameter selection
- [ ] Experiment results tracking and model results visualization
- [ ] Best model selection and approval
- [ ] Transfer of model to higher environments
- [ ] 

# Pipelines:

The pipelines are decoupled. This decoupling controls changes in one pipeline from affecting other pipelines.

## Data

`Trigger: File upload in S3 bucket in raw/` 

* Fetches new data from different sources and keeps raw data in `S3`
* There is some data preparation code which runs. This code is executed via `AWS Lambda`
* After data processing the data is written out again in `S3` to be consumed in training jobs
* Pickles out data transformer class object

## Training

`Trigger: File upload in S3 bucket in processing/`

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

