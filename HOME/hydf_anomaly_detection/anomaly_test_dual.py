# anomaly_classify.py

import cv2, argparse, pickle
import numpy as np
from ultralytics import YOLO

WINDOW_SIZE = 50
FRAME_SKIP = 2

def extract_features_sliding(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    feats = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            res = model.predict(source=frame, verbose=False)[0]
            raw = res.boxes.data.cpu().numpy()
            person_mask = raw[:, 5].astype(int) == 0
            persons = raw[person_mask][:, :4]
            if persons.size > 0:
                x1, y1, x2, y2 = max(persons, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                fw = (x2 - x1) / w
                fh = (y2 - y1) / h
            else:
                cx = cy = fw = fh = np.nan
            feats.append([cx, cy, fw, fh])

        frame_idx += 1

    cap.release()

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

    # Just return the first window for now
    if len(F) >= WINDOW_SIZE:
        return F[:WINDOW_SIZE]
    else:
        return F

def classify_sequence(seq, hmms):
    results = {}
    for name, cfg in hmms.items():
        score = cfg['model'].score(cfg['scaler'].transform(seq))
        margin = score - cfg['threshold']
        results[name] = {'score': score, 'margin': margin, 'threshold': cfg['threshold']}
    best, info = max(results.items(), key=lambda x: x[1]['margin'])
    return (best if info['margin'] >= 0 else "anomaly", results)

def load_hmms(hmm_files):
    hmms = {}
    for label in hmm_files:
        with open(hmm_files[label], 'rb') as f:
            hmms[label] = pickle.load(f)
    return hmms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--walk-model', default='hmm_walk.pkl')
    parser.add_argument('--enter-model', default='hmm_enter.pkl')
    args = parser.parse_args()

    model = YOLO(args.yolo_model)
    seq = extract_features_sliding(args.video, model)

    hmms = load_hmms({'walk': args.walk_model, 'enter': args.enter_model})
    label, scores = classify_sequence(seq, hmms)

    print(f"\nPrediction: {label.upper()}")
    for k, v in scores.items():
        print(f"  {k}: score={v['score']:.2f}, margin={v['margin']:.2f}, threshold={v['threshold']:.2f}")
