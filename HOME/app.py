from flask import Flask, Response, render_template
import cv2, threading, requests, base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import hashlib
import sys, time
import queue
from collections import deque

sys.path.append('..')
from secret import SECRET_KEY, IV_HEX
from hydf_face_recognition.face import identify_face, get_name_label
from hydf_object_detection.object import identify_object, get_object_label
from hydf_anomaly_detection.anomaly import AnomalyDetector, get_normal_behavior

app = Flask(__name__)

# Anomaly detector setup
anom_detector = AnomalyDetector()

# Shared state
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
recording = False
frames = []
FRAME_SIZE = 50
current_name = ""
current_risk_score = 0
object_labels = []
frame_queue = queue.Queue(maxsize=1)
anom_buffer = deque(maxlen=FRAME_SIZE)
votes = deque(maxlen=5)
anom_lock = threading.Lock()
anomaly_label = '...'

# Face recognition thread
def recognition_worker():
    global current_name
    while True:
        try:
            frame = frame_queue.get(timeout=20)
            name = identify_face(frame)
            current_name = name if name else "Unknown"
        except queue.Empty:
            continue

# Anomaly detection thread
def anomaly_worker():
    global anomaly_label
    while True:
        with anom_lock:
            buf = list(anom_buffer)
        if len(buf) == FRAME_SIZE:
            try:
                anomaly_label = anom_detector.detect(buf)
            except Exception as e:
                print("[Anomaly Thread] Detection error:", e)
                anomaly_label = "error"
        time.sleep(0.1)

# Add timestamp to a frame
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
    cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    cv2.putText(frame, timestamp, (text_x, text_y), font, font_scale, font_color, thickness)
    return frame

@app.route('/')
def index():
    return render_template('index.html')

def process_frame(frame, frame_count):
    global anomaly_label, object_labels, current_risk_score

    if frame_count % 30 == 0:
        try:
            frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    with anom_lock:
        anom_buffer.append(frame.copy())

    obj_result = identify_object(frame)

    if len(obj_result.boxes) > 0:
        raw = obj_result.boxes.data.cpu().numpy()
        classes = raw[:, 5].astype(int)
        object_labels = [obj_result.names[c] for c in classes]
    else:
        object_labels = []

    current_risk_score = calculate_risk_score(current_name, object_labels, anomaly_label)

    frame = obj_result.plot()
    cv2.putText(frame, f"Behavior: {anomaly_label.upper()}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    if current_name:
        cv2.putText(frame, f"Name: {current_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return add_timestamp(frame)

def calculate_risk_score(name_label, object_labels, behavior_label):
    names = get_name_label()
    objects = get_object_label()
    normal_behaviors = get_normal_behavior()

    risk_score = 0

    if name_label not in names:
        risk_score += 25

    for object_label in object_labels:
        if object_label == 'box':
            risk_score += 5
        elif object_label == 'gun':
            risk_score += 40
        elif object_label == 'face_half_covered':
            risk_score += 10
        elif object_label == 'face_fully_covered':
            risk_score += 20

    if behavior_label not in normal_behaviors:
        risk_score += 10

    return min(risk_score, 100)

@app.route('/risk_score')
def get_risk_score():
    global current_risk_score
    return {'risk': current_risk_score}

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        frame_count = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = process_frame(frame, frame_count)
            frame_count += 1
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_file/<name>')
def video_file(name):
    def gen_file_frames():
        global anom_buffer, votes, anomaly_label
        with anom_lock:
            anom_buffer.clear()
        votes.clear()
        anomaly_label = "..."

        path = f'static/{name}.mp4'
        video = cv2.VideoCapture(path)

        frame_count = 0
        skip_every = 2

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            if frame_count % skip_every == 0:
                frame = process_frame(frame, frame_count)
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            frame_count += 1

        video.release()

    return Response(gen_file_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    out = cv2.VideoWriter('recorded.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
    for frame in frames:
        out.write(frame)
    out.release()

    with open('recorded.avi', "rb") as file:
        video_data = base64.b64encode(file.read())
        key = hashlib.sha256(SECRET_KEY.encode()).digest()
        iv = bytes.fromhex(IV_HEX)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        encrypted = cipher.encrypt(pad(video_data, AES.block_size))

    with open("encrypted_video.bin", "wb") as f:
        f.write(encrypted)

    with open("encrypted_video.bin", "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
        requests.post('http://localhost:5001/upload', json={'video': encoded})

    frames.clear()
    return "Recording stopped and sent"

def capture_frames():
    global frames, recording, cap
    while recording:
        success, frame = cap.read()
        if success:
            frames.append(add_timestamp(frame))

if __name__ == '__main__':
    threading.Thread(target=recognition_worker, daemon=True).start()
    threading.Thread(target=anomaly_worker, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
