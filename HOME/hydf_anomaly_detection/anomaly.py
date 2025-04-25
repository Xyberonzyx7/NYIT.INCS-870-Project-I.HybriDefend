# hydf_anomaly_detection/hmm_detector.py

import cv2, pickle
import numpy as np
import os
from ultralytics import YOLO
from scipy.ndimage import uniform_filter1d

WINDOW_SIZE = 50
FRAME_SKIP = 2
SMOOTH_KERNEL = 5

module_dir = os.path.dirname(os.path.abspath(__file__))

def get_normal_behavior():
    return ['walk', 'enter']

class AnomalyDetector:
    def __init__(self):
        self.model = YOLO(module_dir + '/yolov8n.pt')
        self.hmms = {}
        for label in ['walk', 'enter']:
            path = os.path.join(module_dir, f'hmm_{label}.pkl')
            with open(path, 'rb') as f:
                self.hmms[label] = pickle.load(f)

    def extract_features(self, frames):
        if not frames:
            return np.zeros((1, 4))  # very short fake input to avoid crash

        w, h = frames[0].shape[1], frames[0].shape[0]
        feats = []
        for i, frame in enumerate(frames):
            if i % FRAME_SKIP != 0:
                continue
            res = self.model.predict(source=frame, verbose=False)[0]
            raw = res.boxes.data.cpu().numpy()
            person_mask = raw[:, 5].astype(int) == 0
            persons = raw[person_mask][:, :4]
            if persons.size > 0:
                x1, y1, x2, y2 = max(persons, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                fw = (x2 - x1) / w
                fh = (y2 - y1) / h
            else:
                cx = cy = fw = fh = np.nan
            feats.append([cx, cy, fw, fh])

        F = np.vstack(feats)

        valid = ~np.isnan(F[:, :4]).any(axis=1)
        if valid.any():
            first = np.argmax(valid)
            F[:first, :4] = F[first, :4]
        else:
            F[:, 0:2] = 0.5
            F[:, 2:4] = 0.0

        for i in range(1, F.shape[0]):
            for j in range(4):
                if np.isnan(F[i, j]):
                    F[i, j] = F[i-1, j]
        for j in range(4):
            F[:, j] = uniform_filter1d(F[:, j], size=SMOOTH_KERNEL, mode='nearest')

        return F


    def detect(self, frames):
        F = self.extract_features(frames)
        results = {}
        for name, cfg in self.hmms.items():
            score = cfg['model'].score(cfg['scaler'].transform(F))
            margin = score - cfg['threshold']
            results[name] = {'score': score, 'margin': margin}
        best = max(results.items(), key=lambda x: x[1]['margin'])
        return best[0] if best[1]['margin'] >= 0 else 'anomaly'
