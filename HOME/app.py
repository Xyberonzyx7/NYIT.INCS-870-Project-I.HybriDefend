import json
from flask import Flask, Response, render_template, request, jsonify, send_from_directory
import cv2, threading, requests, base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import hashlib
import sys, time
import queue
from collections import deque
import boto3
import datetime
import os
import subprocess

sys.path.append('..')
from secret import SECRET_KEY, IV_HEX
from hydf_face_recognition.face import identify_face, get_name_label
from hydf_object_detection.object import identify_object, get_object_label
from hydf_anomaly_detection.anomaly import AnomalyDetector, get_normal_behavior

# Flask app
app = Flask(__name__)

# AWS setup
rekognition_client = boto3.client('rekognition', region_name='us-east-2')
s3_client = boto3.client('s3', region_name='us-east-2')
s3_bucket = 'incs870bucket'
latest_uploaded_key = None
rekognition_job_id = None

# Local shared states
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
object_presence = set()  # ⭐ New: to track objects seen
log_entries = []         # ⭐ New: to track logs
LOG_FOLDER = "static/logs"      # ⭐ New: logs folder

anom_detector = AnomalyDetector()

if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

# --- Threads ---

def recognition_worker():
    global current_name
    while True:
        try:
            frame = frame_queue.get(timeout=20)
            name = identify_face(frame)
            current_name = name if name else "Unknown"
        except queue.Empty:
            continue

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

# --- Utils ---

def add_timestamp(frame):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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

def calculate_risk_score(name_label, object_labels, behavior_label):
    names = get_name_label()
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

def log_event(frame, description):
    def save_log():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{LOG_FOLDER}/{timestamp}.jpg"
        cv2.imwrite(filename, frame)

        # Only after the image is actually saved, then add to log_entries
        log_entry = {
            'time': timestamp,
            'description': description,
            'image': f"static/logs/{timestamp}.jpg"  # ⭐ Make sure path includes static/
        }
        log_entries.append(log_entry)

        # Save the logs.json again
        with open(f"{LOG_FOLDER}/logs.json", "w") as f:
            json.dump(log_entries, f, indent=2)

    threading.Timer(2.0, save_log).start()



# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logs')
def view_logs():
    return render_template('logs.html', logs=reversed(log_entries))


@app.route('/start_recording')
def start_recording():
    global recording, frames
    frames.clear()
    recording = True
    threading.Thread(target=capture_frames).start()
    return "Recording started"

@app.route('/stop_recording')
def stop_recording():
    global recording, frames, latest_uploaded_key, rekognition_job_id
    recording = False

    out = cv2.VideoWriter('temp.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
    for frame in frames:
        out.write(frame)
    out.release()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    final_mp4 = f"{timestamp}.mp4"
    subprocess.run(["ffmpeg", "-i", "temp.avi", "-c:v", "libx264", "-preset", "veryfast", "-movflags", "+faststart", final_mp4])

    s3_client.upload_file(final_mp4, s3_bucket, final_mp4)
    latest_uploaded_key = final_mp4
    print(f"📤 Uploaded {final_mp4} to S3 bucket {s3_bucket}")

    rekognition_response = rekognition_client.start_label_detection(
        Video={'S3Object': {'Bucket': s3_bucket, 'Name': final_mp4}},
        MinConfidence=50,
    )
    rekognition_job_id = rekognition_response['JobId']
    print(f"🚀 Started Rekognition Job ID: {rekognition_job_id}")

    os.remove("temp.avi")
    os.remove(final_mp4)

    return "Recording stopped, uploaded and Rekognition started"

@app.route('/rekognition_result')
def rekognition_result():
    global rekognition_job_id

    if not rekognition_job_id:
        return {'status': 'no_job'}

    result = rekognition_client.get_label_detection(JobId=rekognition_job_id, SortBy='TIMESTAMP')

    if result['JobStatus'] == 'IN_PROGRESS':
        return {'status': 'processing'}

    if result['JobStatus'] == 'FAILED':
        return {'status': 'failed'}

    if result['JobStatus'] == 'SUCCEEDED':
        with open("labels_output.json", "w") as f:
            json.dump(result, f, indent=2)
        print("💾 Saved detailed result to labels_output.json")

    labels = []
    for item in result['Labels']:
        label = item['Label']['Name']
        labels.append(label)

    risk_score = min(len(labels) * 20, 100)

    return {'status': 'done', 'actions': labels, 'risk': risk_score}

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

@app.route('/door_state')
def door_state():
    return jsonify({'state': app.config.get('DOOR_STATE', 'closed')})

def process_frame(frame, frame_count):
    global anomaly_label, object_labels, current_risk_score, object_presence

    if frame_count % 30 == 0:
        try:
            frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    with anom_lock:
        anom_buffer.append(frame.copy())

    obj_result = identify_object(frame)

    detected_objects = set()

    if len(obj_result.boxes) > 0:
        raw = obj_result.boxes.data.cpu().numpy()
        classes = raw[:, 5].astype(int)
        object_labels = [obj_result.names[c] for c in classes]
        detected_objects = set(object_labels)
    else:
        object_labels = set()

    # New appearance logging
    new_objects = detected_objects - object_presence
    for obj in new_objects:
        log_event(frame, f"{obj} appeared")

    # Disappearance logging
    # disappeared_objects = object_presence - detected_objects
    # for obj in disappeared_objects:
    #     log_event(frame, f"{obj} disappeared")
    # Inside process_frame()

    if current_name.lower() == "david" and anomaly_label.lower() == "enter":
        # ⭐ Save a flag that the door should open
        with app.app_context():
            app.config['DOOR_STATE'] = 'open'
    else:
        with app.app_context():
            app.config['DOOR_STATE'] = 'closed'

    object_presence = detected_objects

    current_risk_score = calculate_risk_score(current_name, list(object_labels), anomaly_label)

    frame = obj_result.plot()
    cv2.putText(frame, f"Behavior: {anomaly_label.upper()}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    if current_name:
        cv2.putText(frame, f"Name: {current_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return add_timestamp(frame)

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
