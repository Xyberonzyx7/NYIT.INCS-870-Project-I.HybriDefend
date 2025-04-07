from flask import Flask, request, jsonify
import base64, os
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import hashlib
import datetime
import sys
sys.path.append('..')
from secret import SECRET_KEY, IV_HEX

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload():
    encrypted_video = request.json['video']

    try:
        encrypted_bytes = base64.b64decode(encrypted_video)
        key = hashlib.sha256(SECRET_KEY.encode()).digest()
        iv = bytes.fromhex(IV_HEX)

        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_data = unpad(cipher.decrypt(encrypted_bytes), AES.block_size)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_filename = f"{timestamp}.avi"
        video_data = base64.b64decode(decrypted_data)
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)

        with open(video_path, "wb") as f:
            f.write(video_data)

        return jsonify({'status': 'success'}), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
