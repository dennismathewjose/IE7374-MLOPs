import boto3
from app import config

def upload_to_s3(file_path):
    s3 = boto3.client(
        's3',
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
    )

    s3.upload_file(file_path, config.S3_BUCKET_NAME, file_path)
    print(f"File uploaded to s3://{config.S3_BUCKET_NAME}/{file_path}")
