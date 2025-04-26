import boto3
from botocore.exceptions import NoCredentialsError

# --- Config ---
bucket_name = 'incs870bucket'   # REPLACE with your bucket
file_path   = 'deliver_fixed.mp4'        # REPLACE with your file path
s3_key      = 'deliver_fixed.mp4'      # Name to store in S3

# --- Upload ---
s3 = boto3.client('s3')

try:
    s3.upload_file(file_path, bucket_name, s3_key)
    print(f"✅ Uploaded {file_path} to s3://{bucket_name}/{s3_key}")
except FileNotFoundError:
    print("❌ File not found.")
except NoCredentialsError:
    print("❌ AWS credentials not configured.")
