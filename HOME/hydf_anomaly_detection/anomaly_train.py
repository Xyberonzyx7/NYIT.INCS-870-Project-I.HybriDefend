#!/usr/bin/env python3
"""
YOLOv8 (face+parcel) + HMM Behavioral Classifier & Anomaly Detector

Pipeline:
1. Load custom YOLO model that detects face and parcel
2. Extract per-frame features: [face_cx, face_cy, face_w, face_h, parcel_flag]
3. Clean (backfill & forward-fill) missing values
4. Train one Gaussian HMM per behavior: walk, enter, deliver
5. Save each HMM + scaler + threshold
6. Classify new clips or flag anomalies

CMD:
python anomaly_train.py --yolo-model best.pt --face-class 2 --parcel-class 0 --walk-dir 1_walkby --enter-dir 2_entry --deliver-dir 3_package_deliver --test test_deliver

"""
import os
import argparse
import pickle

import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm


def extract_features(video_path, model, face_cls, parcel_cls):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.array([[0.5, 0.5, 0.0, 0.0, 0]], dtype=float)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    feats = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = model.predict(source=frame, verbose=False)[0]
        raw = res.boxes.data.cpu().numpy()  # [x1,y1,x2,y2,conf,cls]
        # face
        mask_f = raw[:, 5].astype(int) == face_cls
        faces = raw[mask_f][:, :4]
        # parcel flag
        parcel_flag = 1 if np.any(raw[:, 5].astype(int) == parcel_cls) else 0
        if faces.size > 0:
            x1, y1, x2, y2 = max(faces, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            fw = (x2 - x1) / w
            fh = (y2 - y1) / h
        else:
            cx = cy = fw = fh = np.nan
        feats.append([cx, cy, fw, fh, parcel_flag])
    cap.release()
    if len(feats) == 0:
        return np.array([[0.5, 0.5, 0.0, 0.0, 0]], dtype=float)
    F = np.vstack(feats).astype(float)
    # backfill initial NaNs
    valid = ~np.isnan(F[:, :4]).any(axis=1)
    if valid.any():
        first = np.argmax(valid)
        F[:first, :4] = F[first, :4]
    else:
        F[:, 0:2] = 0.5
        F[:, 2:4] = 0.0
    # forward-fill
    for i in range(1, F.shape[0]):
        for j in range(4):
            if np.isnan(F[i, j]):
                F[i, j] = F[i-1, j]
    return F


def train_hmm(seqs, n_states, percentile):
    lengths = [s.shape[0] for s in seqs]
    X = np.vstack(seqs)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type='full',
        n_iter=100,
        tol=1e-4,
        verbose=False
    )
    model.fit(Xs, lengths)
    lls = [model.score(scaler.transform(s)) for s in seqs]
    thr = np.percentile(lls, percentile)
    return {'model': model, 'scaler': scaler, 'threshold': thr}


def save_hmm(cfg, path):
    with open(path, 'wb') as f:
        pickle.dump(cfg, f)


def load_hmms(paths):
    hmms = {}
    for name, p in paths.items():
        with open(p, 'rb') as f:
            hmms[name] = pickle.load(f)
    return hmms


def classify_sequence(seq, hmms):
    results = {}
    for name, cfg in hmms.items():
        score = cfg['model'].score(cfg['scaler'].transform(seq))
        margin = score - cfg['threshold']
        results[name] = {'score': score, 'threshold': cfg['threshold'], 'margin': margin}
    best, info = max(results.items(), key=lambda x: x[1]['margin'])
    return (best if info['margin'] >= 0 else 'anomaly', results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--yolo-model', required=True, help='.pt file that detects face & parcel')
    p.add_argument('--face-class', type=int, required=True)
    p.add_argument('--parcel-class', type=int, required=True)
    p.add_argument('--walk-dir', required=True)
    p.add_argument('--enter-dir', required=True)
    p.add_argument('--deliver-dir', required=True)
    p.add_argument('--test', nargs='+', required=True)
    p.add_argument('--states', type=int, default=7)
    p.add_argument('--percentile', type=float, default=5.0)
    p.add_argument('--out-prefix', default='hmm')
    args = p.parse_args()

    print(f"Loading YOLO model: {args.yolo_model}")
    model = YOLO(args.yolo_model)
    dirs = {'walk': args.walk_dir, 'enter': args.enter_dir, 'deliver': args.deliver_dir}
    mp = {}
    # train HMMs
    for name, d in dirs.items():
        seqs = []
        print(f"Training HMM for {name}...")
        for f in sorted(os.listdir(d)):
            path = os.path.join(d, f)
            if os.path.isfile(path):
                seq = extract_features(path, model, args.face_class, args.parcel_class)
                seqs.append(seq)
                print(f"  {f}: {seq.shape[0]} frames")
        cfg = train_hmm(seqs, args.states, args.percentile)
        out = f"{args.out_prefix}_{name}.pkl"
        save_hmm(cfg, out)
        mp[name] = out
        print(f"  Saved -> {out} (thr={cfg['threshold']:.2f})")

    # inference
    print("Classifying test videos...")
    hmms = load_hmms(mp)
    for t in args.test:
        seq = extract_features(t, model, args.face_class, args.parcel_class)
        label, scores = classify_sequence(seq, hmms)
        stats = ', '.join(f"{k}:{v['score']:.1f}" for k,v in scores.items())
        print(f"{t}: {label.upper()} ({stats})")

if __name__ == '__main__':
    main()
