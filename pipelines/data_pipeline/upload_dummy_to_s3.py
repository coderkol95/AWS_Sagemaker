import boto3

if __name__=="__main__":

    s3=boto3.client("s3")
    s3.upload_file('dummy.txt',"dataforml",'raw/dummy.txt')