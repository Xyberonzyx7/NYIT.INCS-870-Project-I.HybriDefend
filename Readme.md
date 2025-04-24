# ReadMe

### Environment Setup

- GLOBAL
	#### Files:
	- secret.py
		```python
		SECRET_KEY = "anythinghere"
		IV_HEX = "01100001000111010010001000100000"  # 16 bytes = 128 bitsIV = b
		```

- HOME (using python 3.10)
	- py -3.10 venv venv_py310
	- .\venv_py310\Scripts\Activate.ps1
	- python app.py
	- deactivate

	#### Libraries
	- pip install flask
	- pip install opencv-python
	- pip install requests
	- pip install pycryptodome
	- pip install tensorflow
	- pip install face_recognition
	- pip install numpy
	- pip install pillow
	- pip install ultralytics


	#### Files
	- HOME/hydf_face_recognition/faces/
    	- {Username1}/{Images}
    	- {Username2}/{Images}

- CLOUD (using python 3.10)
	- py -3.10 venv venv_py310
	- .\venv_py310\Scripts\Activate.ps1
	- python cloud_app.py
	- deactivate

	#### Libraries
	- pip install pycryptodome

### Learn
- requirements.txt
	- pip freeze > requirements.txt
	- pip install -r requirements.txt
