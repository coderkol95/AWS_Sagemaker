#https://www.youtube.com/watch?v=iuqIusA_JNA

# Pipeline steps
## 1. Create role in IAM to allow S3 read-write access
    # Created a role for all S3 access. Need more fine grained access ideally

## 2. Trigger step: Read entries in raw/ in `S3`
    # This was easy. Added prefix

## 3. Execution step: Run data preparation pipeline and save the pipeline pickle object to artifacts/ in `S3`
    # Need to do this via containers as native Lambda functionality is very primitive.

## 4. Write-out step: Write modified files to processed/ in `S3` and pickle file to artifacts in `S3`

import boto3



# Everything is done by lambda_handler.py in docker_files/data_pipeline