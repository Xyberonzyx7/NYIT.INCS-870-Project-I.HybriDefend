from ultralytics import YOLO
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

# Replace your Keras load_model line with this:
yolo_model = YOLO(module_dir + '/best.pt')

objects = ['box', 'gun', 'face', 'face_half_covered', 'face_fully_covered']

def get_object_label():
    return objects


def identify_object(frame):
    # object detection
    results = yolo_model(frame, imgsz=640, conf=0.70, verbose=False)[0]

    return results
