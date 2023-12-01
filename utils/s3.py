import boto3
import os

with open(".keys.txt",'r') as f:
    AK=f.readline()[:-1] # To remove the newline character
    SK=f.readline()[:-1] # To remove the newline character
    S3_REGION=f.readline()

BUCKET_NAME="dataforml"

s3 = boto3.client(
's3',
aws_access_key_id=AK,
aws_secret_access_key=SK,
region_name=S3_REGION
)

def upload(file_path:str,
           file_name:str=None):
    
    if file_path is not None and file_name is None:
        file_name="raw/"+file_path.split("/")[-1]
    
    elif file_path is None:
        raise ValueError("Please enter a valid file path.")
    
    try:
        s3.upload_file(file_path,BUCKET_NAME,file_name)
        return 1
    except:
        return 0

def download(file_name:str,
           file_path:str=None):
    
    if file_name is not None and file_path is None:
        if not os.path.exists('./downloads/processed/'):
            os.makedirs('./downloads/processed/')
        file_path='./downloads/processed/'+file_name
        
    elif file_name is None:
        raise ValueError("Please enter a valid file name.")
    
    try:
        s3.download_file(BUCKET_NAME,file_name,file_path)
        return 1
    except:
        return 0
