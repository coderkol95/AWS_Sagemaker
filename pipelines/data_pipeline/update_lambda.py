import boto3
import os
import json
config=json.load(open(".keys.json"))
AWS_ACC_ID = config["AWS_ACC_ID"]

def update_lambda_fn(image_uri):

    lambda_fn=boto3.client('lambda')
    lambda_fn.update_function_code(
    FunctionName="data-pipeline",
    ImageUri=image_uri,
    Publish=True
    )

if __name__=="__main__":

    image=f"{AWS_ACC_ID}.dkr.ecr.us-east-1.amazonaws.com/data-pipeline:{os.environ['IMAGE_TAG']}"
    update_lambda_fn(image)