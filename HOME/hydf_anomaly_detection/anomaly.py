#!/usr/bin/env python3
import os
import argparse
import pickle

import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

module_dir = os.path.dirname(os.path.abspath(__file__))

# --- SHARED FEATURE EXTRACTION HELPERS ---

def _extract_from_frame(frame, model, face_cls, parcel_cls):
    """
    Extract features [cx, cy, fw, fh, parcel_flag] from a single BGR frame.
    """
    h, w = frame.shape[:2]
    res = model.predict(source=frame, verbose=False)[0]
    raw = res.boxes.data.cpu().numpy()  # [x1,y1,x2,y2,conf,cls]

    # face detection
    mask_f = raw[:, 5].astype(int) == face_cls
    faces = raw[mask_f][:, :4]
    # parcel presence
    parcel_flag = int((raw[:, 5].astype(int) == parcel_cls).any())

    if faces.size:
        x1, y1, x2, y2 = max(faces, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        fw = (x2 - x1) / w
        fh = (y2 - y1) / h
    else:
        cx = cy = fw = fh = np.nan

    return [cx, cy, fw, fh, parcel_flag]


def _clean_sequence(F):
    """
    Backfill and forward-fill NaNs in the first four columns of sequence F (shape T x 5).
    """
    if F.size == 0:
        return np.array([[0.5, 0.5, 0, 0, 0]], dtype=float)
    valid = ~np.isnan(F[:, :4]).any(axis=1)
    if valid.any():
        first = np.argmax(valid)
        F[:first, :4] = F[first, :4]
    else:
        F[:, :2] = 0.5
        F[:, 2:4] = 0
    for i in range(1, len(F)):
        for j in range(4):
            if np.isnan(F[i, j]):
                F[i, j] = F[i-1, j]
    return F


def extract_features_from_video(video_path, model, face_cls, parcel_cls):
    """
    Build feature sequence from a video file on disk.
    """
    cap = cv2.VideoCapture(video_path)
    feats = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        feats.append(_extract_from_frame(frame, model, face_cls, parcel_cls))
    cap.release()
    return _clean_sequence(np.array(feats, dtype=float))


def extract_features_from_frames(frames, model, face_cls, parcel_cls):
    """
    Build feature sequence from an in-memory list of BGR frames.
    """
    feats = []
    for frame in frames:
        feats.append(_extract_from_frame(frame, model, face_cls, parcel_cls))
    return _clean_sequence(np.array(feats, dtype=float))

# --- HMM MODEL LOADING & CLASSIFICATION ---

def load_models():
    """
    Load walk.pkl, enter.pkl, deliver.pkl from module directory.
    Returns dict of name -> {'model', 'scaler', 'threshold'}.
    """
    hmms = {}
    for name in ('walk', 'enter', 'deliver'):
        path = os.path.join(module_dir, f"{name}.pkl")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Cannot find {path}")
        with open(path, 'rb') as f:
            hmms[name] = pickle.load(f)
    return hmms


def classify_sequence(seq, hmms):
    """
    Score seq against each HMM, return (label, results) where results is dict name -> {'score','margin'}.
    """
    results = {}
    for name, cfg in hmms.items():
        score = cfg['model'].score(cfg['scaler'].transform(seq))
        margin = score - cfg['threshold']
        results[name] = {'score': score, 'margin': margin}
    best, info = max(results.items(), key=lambda x: x[1]['margin'])
    label = best if info['margin'] >= 0 else 'anomaly'
    return label, results

# --- DETECTOR CLASS FOR FLASK INTEGRATION ---

class AnomalyDetector:
    def __init__(self, face_cls, parcel_cls):
        self.yolo       = YOLO(module_dir + "/best.pt")
        self.face_cls   = face_cls
        self.parcel_cls = parcel_cls
        self.hmms       = load_models()

    def detect(self, frames):
        """
        frames: list of BGR numpy arrays
        returns: 'walk', 'enter', 'deliver', or 'anomaly'
        """
        seq, _ = classify_sequence(
            extract_features_from_frames(
                frames, self.yolo, self.face_cls, self.parcel_cls
            ),
            self.hmms
        )
        return seq

# --- MAIN FUNCTION FOR SCRIPT USAGE ---

def main():
    parser = argparse.ArgumentParser(
        description='Anomaly detection on video files using HMMs'
    )
    parser.add_argument('--face-class', type=int, required=True,
                        help='Class index of face in YOLO model.names')
    parser.add_argument('--parcel-class', type=int, required=True,
                        help='Class index of parcel')
    parser.add_argument('--test', nargs='+', required=True,
                        help='Video file(s) to classify')
    args = parser.parse_args()

    # Load YOLO for file-based extraction
    yolo = YOLO(module_dir + "/best.pt")
    hmms = load_models()

    for vid in args.test:
        seq = extract_features_from_video(
            vid, yolo, args.face_class, args.parcel_class
        )
        label, results = classify_sequence(seq, hmms)
        stats = ', '.join(f"{k}:{v['score']:.1f}" for k,v in results.items())
        print(f"{vid}: {label.upper()} [{stats}]")

if __name__ == '__main__':
    main()