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
import queue
from collections import deque, Counter
import threading, time


from hydf_face_recognition.face import identify_face
from hydf_object_detection.object import identify_object
from hydf_anomaly_detection.anomaly import AnomalyDetector

# configure once at startup
anom_detector = AnomalyDetector(
    face_cls=2,
    parcel_cls=0,
)

clip_buffer = []

app = Flask(__name__)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
recording = False
frames = []
frame_queue = queue.Queue(maxsize=1)  # Queue to hold frames for recognition
current_name = ""  # The name identified by the recognition thread

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():

    def gen_frames():
        frame_count = 0
        face_process_interval = 30
        global cap, clip_buffer
        while True:
            success, frame = cap.read()
            if not success:
                break


            # Face recognition
            if frame_count % face_process_interval == 0:
                # copy so the background thread sees the right image
                try:
                    frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass

            
            # append the incoming frame to the shared buffer
            with anom_lock:
                anom_buffer.append(frame.copy())


            # # object detection + draw object
            frame = identify_object(frame)

            # immediately overlay the last known majority label
            cv2.putText(
                frame,
                f"Behavior: {anomaly_label.upper()}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            # draw the last computed label:
            cv2.putText(frame,
                        f"Behavior: {anomaly_label.upper()}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # draw name
            if current_name:
                # print(current_name)
                cv2.putText(frame, f"Name: {current_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # draw timestamp
            frame = add_timestamp(frame)

            frame_count += 1

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

@app.route('/video_walk')
def video_walk():
    return send_file('test_walk.mp4', mimetype='video/mp4')

@app.route('/video_entry')
def video_entry():
    return send_file('test_entry.mp4', mimetype='video/mp4')

@app.route('/video_deliver')
def video_deliver():
    return send_file('test_deliver.mp4', mimetype='video/mp4')


"""
Face Recognition Functions
Face Recognition Functions
Face Recognition Functions
"""
def recognition_worker():
    """
    The recognition thread. Pulls frames from the queue and processes them.
    """
    global current_name
    while True:
        try:
            # Get the most recent frame
            frame = frame_queue.get(timeout=20)
            # Run face recognition
            name = identify_face(frame)
            current_name = name if name else "Unknown"
        except queue.Empty:
            continue

"""
Anomaly Detection Functions
Anomaly Detection Functions
Anomaly Detection Functions
"""

# ——— Configuration ———
WINDOW_SIZE     = 100   # how many frames in each window
VOTE_SIZE       = 1    # how many window‐labels to keep for voting
DETECT_INTERVAL = 0.1  # seconds between worker runs

# ——— Shared state ———
anom_buffer   = deque(maxlen=WINDOW_SIZE)
votes         = deque(maxlen=VOTE_SIZE)
anomaly_label = '…'
anom_lock     = threading.Lock()

def anomaly_worker():
    global anomaly_label
    while True:
        # 1) copy and clear buffer under lock
        with anom_lock:
            buf = list(anom_buffer)

        # 2) if we have a full window, classify it
        if len(buf) == WINDOW_SIZE:
            label = anom_detector.detect(buf)
            # votes.append(label)

            # 3) compute the majority vote
            # majority = Counter(votes).most_common(1)[0][0]
            anomaly_label = label

        time.sleep(DETECT_INTERVAL)

"""
Main Function
Main Function
Main Function
"""

if __name__ == '__main__':

    # Start the recognition thread before running the app
    recognition_thread = threading.Thread(target=recognition_worker)
    recognition_thread.daemon = True
    recognition_thread.start()

    # anomaly detection thread
    threading.Thread(target=anomaly_worker, daemon=True).start()



    app.run(host='0.0.0.0', port=5000)
