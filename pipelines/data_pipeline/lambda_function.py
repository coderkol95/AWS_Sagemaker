import boto3
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import sys
import logging
logger = logging.getLogger()
 
BUCKET_NAME="dataforml"

def get_files_from_s3(s3,bucket_name=BUCKET_NAME):

    s3.download_file(bucket_name,"raw/X.csv","/tmp/X.csv")
    X = pd.read_csv("/tmp/X.csv")
    return X

def transform_data(X):

    s=StandardScaler()
    X_trans = s.fit_transform(X)
    return s,X_trans

def save_files_locally(X_trans,s):
    pd.DataFrame(X_trans).to_csv("/tmp/X_trans.csv")
    with open("/tmp/scaler.pkl","wb") as f:
        pickle.dump(s,f)

def put_files_to_s3(s3,
                    X_loc,
                    data_pipe_loc,
                    bucket_name=BUCKET_NAME):
    
    s3.upload_file(X_loc,bucket_name,"preprocessed/X.csv")
    s3.upload_file(data_pipe_loc,bucket_name,"artifacts/scaler.pkl")

def handler(event, context):

    s3 = boto3.client("s3")
    logger.info("started")
    X=get_files_from_s3(s3)
    logger.info("X received")
    s,X_trans=transform_data(X)
    logger.info("transformed")
    save_files_locally(X_trans,s)
    logger.info("Saved locally")
    put_files_to_s3(s3,"/tmp/X_trans.csv","/tmp/scaler.pkl")  
    logger.info("Written out to s3")  


    
