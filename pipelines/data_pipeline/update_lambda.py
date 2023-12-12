import boto3
import os

def update_lambda_fn(image_uri):

    lambda_fn=boto3.client('lambda')
    lambda_fn.update_function_code(
    FunctionName="data-pipeline",
    ImageUri=image_uri,
    Publish=True
    )

if __name__=="__main__":

    image=f"{os.environ['ECR_REGISTRY']}/{os.environ['ECR_REPOSITORY']}:{os.environ['IMAGE_TAG']}"

    update_lambda_fn(image)

