# Readme

1. Create bucket
   ![picture 0](assets/2025-04-25-20-04-13.png)  
2. Create IAM user
   ![picture 1](assets/2025-04-25-20-10-50.png)  
   Permission:  
   AmazonRekognitionFullAccess  
   AmazonS3ReadOnlyAccess
   ![picture 2](assets/2025-04-25-20-12-41.png)  
   ![picture 3](assets/2025-04-25-20-13-21.png)  
3. Create Access Key  
   ![picture 4](assets/2025-04-25-20-16-20.png)  
   ![picture 5](assets/2025-04-25-20-18-00.png)  
   ![picture 6](assets/2025-04-25-20-19-05.png)  
   ![picture 7](assets/2025-04-25-20-20-40.png)  

4. Configure AWS credentials locally  
   
   Installing or updating to the latest version of the AWS CLI  
   [download](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)  
   ![picture 10](assets/2025-04-25-20-35-58.png)  

5. Let IAM user (rekonition-test-user) to be able to access the incs870bucket  
   ![picture 8](assets/2025-04-25-20-33-39.png)  
   ![picture 9](assets/2025-04-25-20-34-19.png)  

6. Upload the video by 'rekonition-test-user'
   ```python
   python upload_video.py
   ```
   ![picture 12](assets/2025-04-25-20-37-21.png)  
   ![picture 11](assets/2025-04-25-20-37-09.png)  

7. Setting bucket permission
   ![picture 13](assets/2025-04-25-20-45-58.png)  
   ![picture 14](assets/2025-04-25-20-46-32.png)  

8. change video format (Even though the file is .mp4, the inside codec (like audio/video compression format) might not be H.264 — AWS Rekognition only supports certain formats.)
	- Use ffmpeg to change format
	```python
	choco install ffmpeg
	ffmpeg -i deliver.mp4 -c:v libx264 -c:a aac -strict experimental deliver_fixed.mp4
	```

9. Ask AWS Rekonition to analyze my video
   ```python
   python analyze_video.py
   ```

   Response:  
    [8.0s] Surprised (51.0%)  
	[8.0s] Package Delivery (69.4%)  
	[8.0s] Person (86.4%)  
	[8.0s] Package Delivery (69.4%)  
	[8.0s] Package Delivery (69.4%)  
	[8.0s] Person (86.4%)  
	[8.0s] Surprised (51.0%)  
	[8.0s] Teen (59.8%)  
	[8.5s] Box (97.8%)  
	[8.5s] Boy (87.9%)  
	[8.5s] Cardboard (91.5%)  
	[8.5s] Carton (91.5%)  
	[8.5s] Face (75.0%)  
	[8.5s] Head (75.8%)  
	[8.5s] Male (91.3%)  
	[8.5s] Opening Present (55.4%)  
	[8.5s] Package (87.4%)  
	[8.5s] Package Delivery (83.5%)  
	[8.5s] Person (94.3%)  
	[8.5s] Teen (87.9%)  
	💾 Saved detailed result to labels_output.json


   
