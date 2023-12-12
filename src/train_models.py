# Define IAM role
import boto3
import re
import os
import sagemaker
import json

sess = sagemaker.Session()
config=json.load(open("../.keys.json"))
role = config["SAGEMAKER_ARN"] # Also added EC2 access for accessing custom image

# Data should be downloaded inside docker container.

image = config["TRAINING_IMAGE"]

nn_model = sagemaker.estimator.Estimator(
    base_job_name="nn-model-dummy",
    image_uri=image,
    role=role,
    instance_count=1,
    instance_type="ml.m4.xlarge",
    output_path=f"s3://dataforml/model-training/output",
    sagemaker_session=sess,
    input_mode="File",
    hyperparameters={
        "epochs":2
    }
)

training_input=sagemaker.estimator.TrainingInput(s3_data="s3://dataforml/preprocessed/")
nn_model.fit({"train":training_input})