# Hybrid Defend

## Environmental Setup

**Install Anaconda**

It makes managing Python versions, libraries, and virtual environments easy

```cmd
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o .\miniconda.exe

start /wait "" .\miniconda.exe /S

del .\miniconda.exe
```
[anaconda install instructions](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)

**Create & activate python virtual environment**

```cmd
conda create -n tfenv python=3.11

conda activate tfenv
```
**Install dependencies**

```cmd
pip install -r requirements.txt
```

## Gamil Notification Setup

**Gmail App Password**

1. Go to [Google Account Security](https://myaccount.google.com/security).
2. Search **App Passwords**.
3. Provide a new app name
4. Google will generate a 16-character password. Copy it.
5. Create and add your credentials `{project}/email_config.py`

```python
EMAIL_ADDRESS = "your_email@gmail.com"
EMAIL_PASSWORD = "new_generated_app_password"
```

## Facial Recognition Setup

### Folder Layout

Create this structure inside your project:

```
NYIT.INCS-870-Project-I.HybriDefend/
└─ hydf_face_recognition/
   ├─ face.py
   └─ faces/          # <-- Create this folder
      ├─ David/
      │  ├─ img1.jpg
      │  ├─ img2.jpg
      │  └─ ...
      └─ Kevin/
         ├─ img1.jpg
         ├─ img2.jpg
         └─ ...
```

* Put **multiple clear photos** of **David** inside `faces/David/`.
* Put **multiple clear photos** of **Kevin** inside `faces/Kevin/`.
* The current code is set up only for these two people.

### Create Face Encodings

Run:

```bash
python hydf_face_recognition/face.py
```

This will:

* Process the images in `faces/David/` and `faces/Kevin/`.
* Generate `known_faces.json` with stored encodings.
* Perform identify test on `test_face.png`.

### Notes

* If you replace the photos of David or Kevin, delete `known_faces.json` and run again to regenerate encodings.

## Object Detection Setup

* No additional setup is required.

### Model

* **Framework**: [Ultralytics YOLOv8](https://docs.ultralytics.com)
* **Model file**: `hydf_object_detection/best.pt` (already included in the project)
* **Classes supported**:

  * `box`
  * `gun`
  * `face`
  * `face_half_covered`
  * `face_fully_covered`

## AWS S3 Setup (Cloud Service)

* Please refer to [AWS S3 Setup](hydf_aws_rekognition/Readme.md)
