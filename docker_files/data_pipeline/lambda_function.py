import boto3
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

BUCKET_NAME="dataforml"

def get_files_from_s3(s3,bucket_name=BUCKET_NAME):

    s3.download(bucket_name,"raw/X.csv","X.csv")
    X = pd.read_csv("X.csv")
    return X

def transform_data(X):

    s=StandardScaler()
    X_trans = s.fit_transform(X)
    return s,X_trans

def save_files_locally(X_trans,s):
    pd.DataFrame(X_trans).to_csv("X_trans.csv")
    with open("scaler.pkl","wb") as f:
        pickle.dump(s,f)

def put_files_to_s3(s3,
                    X_loc,
                    data_pipe_loc,
                    bucket_name=BUCKET_NAME):
    
    s3.upload(X_loc,bucket_name,"preprocessed/X.csv")
    s3.upload(data_pipe_loc,bucket_name,"artifacts/scaler.pkl")

def handler(event, context):

    s3 = boto3.client("s3")
    X=get_files_from_s3(s3)
    s,X_trans=transform_data(X)
    save_files_locally(X_trans,s)
    put_files_to_s3(s3,"X_trans.csv","scaler.pkl")    


    
