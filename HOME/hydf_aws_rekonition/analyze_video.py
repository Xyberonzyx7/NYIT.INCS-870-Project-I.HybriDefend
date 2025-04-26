import boto3
import time
import json

# 👇 Configuration
bucket = "incs870bucket"    # your bucket name
video_file = "deliver_fixed.mp4"  # your uploaded file name
region = "us-east-2"        # your region

rekognition = boto3.client('rekognition', region_name=region)

# Start analyzing
response = rekognition.start_label_detection(
    Video={'S3Object': {'Bucket': bucket, 'Name': video_file}},
    MinConfidence=50,
)

job_id = response['JobId']
print(f"🚀 Started label detection job: {job_id}")

# Wait for completion
print("⏳ Waiting for Rekognition to complete... (this may take 30-90 seconds)")

while True:
    result = rekognition.get_label_detection(JobId=job_id)
    status = result['JobStatus']
    if status == 'SUCCEEDED':
        print("🎉 Analysis complete!")
        break
    elif status == 'FAILED':
        print("❌ Analysis failed. Full response:")
        print(json.dumps(result, indent=2))
        raise Exception("Video analysis failed.")
    time.sleep(5)

# Output detections
for label_detection in result['Labels']:
    label = label_detection['Label']
    timestamp = label_detection['Timestamp'] / 1000  # ms → seconds
    print(f"[{timestamp:.1f}s] {label['Name']} ({label['Confidence']:.1f}%)")

# Save raw output
with open("labels_output.json", "w") as f:
    json.dump(result, f, indent=2)
print("💾 Saved detailed result to labels_output.json")
