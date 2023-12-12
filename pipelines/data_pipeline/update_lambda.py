import boto3
import argparse

def update_lambda_fn(image_uri):

    lambda_fn=boto3.client('lambda')
    lambda_fn.update_function_code(
    FunctionName="data-pipeline",
    ImageUri=image_uri,
    Publish=True
    )

if __name__=="__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("--image",required=True,type=str)
    image=parser.parse_args()
    update_lambda_fn(image)

