from flask import Flask, Response, render_template
import cv2, threading, requests, base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import hashlib
import sys
sys.path.append('..')  # go up to project root
from secret import SECRET_KEY, IV_HEX

app = Flask(__name__)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
recording = False
frames = []

def capture_frames():
    global frames, recording, cap
    while recording:
        success, frame = cap.read()
        if success:
            frames.append(frame)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        global cap
        while True:
            success, frame = cap.read()
            if not success: break
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording')
def start_recording():
    global recording, frames
    frames.clear()
    recording = True
    threading.Thread(target=capture_frames).start()
    return "Recording started"

@app.route('/stop_recording')
def stop_recording():
    global recording, frames
    recording = False

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_name = 'recorded.avi'
    encrypted_file_name = 'encrypted_video.bin'
    out = cv2.VideoWriter(video_name, fourcc, 20.0, (640, 480))

    for frame in frames:
        out.write(frame)
    out.release()

    # Encrypt the recorded video
    with open(video_name, "rb") as file:
        video_data = base64.b64encode(file.read())
        key = hashlib.sha256(SECRET_KEY.encode()).digest()
        iv = bytes.fromhex(IV_HEX)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        encrypted = cipher.encrypt(pad(video_data, AES.block_size))

    # Save encrypted video locally
    with open(encrypted_file_name, "wb") as enc_file:
        enc_file.write(encrypted)

    # Upload encrypted video to Cloud Server
    with open(encrypted_file_name, "rb") as enc_file:
        encoded_video = base64.b64encode(enc_file.read()).decode()
        requests.post('http://localhost:5001/upload', json={'video': encoded_video})

    frames.clear()
    return "Recording stopped, encrypted locally, and sent to cloud."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
