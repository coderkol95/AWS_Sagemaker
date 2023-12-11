# MNIST on SageMaker with PyTorch Lightning
import json
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

sagemaker_session = sagemaker.Session()
bucket = 's3://dataforml/preprocessed'
role = 'arn:aws:iam::879444378775:role/SagemakerLocalAccess'

# Creates a new PyTorch Estimator with params
estimator = PyTorch(
    image_uri='763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker',
    entry_point='model.py',
    source_dir='src',
    role=role,
    framework_version='1.7',
    py_version='py3',
    train_instance_count=1,
    train_instance_type='ml.m4.xlarge',
    hyperparameters={
        'epochs': 2,
    })

estimator.fit()