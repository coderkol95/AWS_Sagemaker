import boto3

with open(".keys.txt",'r') as f:
    AK=f.readline()[:-1] # To remove the newline character
    SK=f.readline()[:-1] # To remove the newline character
    S3_REGION=f.readline()

BUCKET_NAME="ml-data-for-mlops"

s3 = boto3.client(
's3',
aws_access_key_id=AK,
aws_secret_access_key=SK,
region_name=S3_REGION
)

def upload(bucket_name:str=BUCKET_NAME,
           file_path:str,
           file_name:str):
    
    if file_path is not None and file_name is None:
        file_name=file_path.split("/")[-1]
    
    try:
        s3.upload_file(file_path,bucket_name,file_name)
        return 1
    except:
        return 0

def download(bucket_name:str=BUCKET_NAME,
           file_name:str,
           file_path:str):
    
    try:
        s3.download_file(bucket_name)
        return 1
    except:
        return 0


# return s3, BUCKET_NAME

if __name__=="__main__":

    pass