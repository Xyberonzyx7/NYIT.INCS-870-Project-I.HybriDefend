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