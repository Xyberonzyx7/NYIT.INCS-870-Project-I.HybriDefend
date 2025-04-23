from flask import Flask, Response, render_template
import cv2, threading, requests, base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import hashlib
import sys
sys.path.append('..')  # go up to project root
from secret import SECRET_KEY, IV_HEX
from tensorflow.keras.models import load_model
import numpy as np

# Load your trained model
model = load_model('best_model.keras')

# Define the expected clip length and target frame size
CLIP_LENGTH = 30
TARGET_SIZE = (256, 256)  # Must match training IMAGE_HEIGHT and IMAGE_WIDTH
clip_buffer = []

app = Flask(__name__)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
recording = False
frames = []

def capture_frames():
    global frames, recording, cap
    while recording:
        success, frame = cap.read()
        if success:
            frames.append(add_timestamp(frame))

def add_timestamp(frame):
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    height, width, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 255, 255)
    thickness = 2
    text_size, _ = cv2.getTextSize(timestamp, font, font_scale, thickness)
    text_x = width - text_size[0] - 10
    text_y = height - 10

    # Background rectangle
    cv2.rectangle(frame,
                  (text_x - 5, text_y - text_size[1] - 5),
                  (text_x + text_size[0] + 5, text_y + 5),
                  (0, 0, 0), -1)
    # Timestamp text
    cv2.putText(frame, timestamp, (text_x, text_y), font, font_scale, font_color, thickness)
    return frame

def anomaly_detection(frame):
    # Resize frame to match model input
    resized_frame = cv2.resize(frame, TARGET_SIZE)
    # Use the resized frame directly (training used unnormalized frames)
    clip_buffer.append(resized_frame)
    if len(clip_buffer) > CLIP_LENGTH:
        clip_buffer.pop(0)

    prediction_text = ""
    if len(clip_buffer) == CLIP_LENGTH:
        # Prepare input with shape: (1, CLIP_LENGTH, height, width, channels)
        clip_array = np.array(clip_buffer)
        clip_array = np.expand_dims(clip_array, axis=0)
                
        # Run model prediction
        prediction = model.predict(clip_array)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        # Map index to class name
        CLASS_LIST = ["normal", "shooting"]
        predicted_class = CLASS_LIST[class_idx]
        prediction_text = f"{predicted_class} ({confidence*100:.1f}%)"

    # Overlay prediction text on the frame
    if prediction_text:
        cv2.putText(frame, prediction_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return frame



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        global cap, clip_buffer
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Add timestamp to the frame
            frame = add_timestamp(frame)

            # anomaly detection
            frame = anomaly_detection(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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
