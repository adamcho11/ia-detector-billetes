#!/usr/bin/env python3
"""
Pipeline de Entrenamiento — ORB+FLANN+Homografia + Random Forest (OpenCV).
Sin dependencia de scikit-learn. Usa cv2.ml.RTrees.

Uso:
    python train_pipeline.py --templates ./templates/ --autenticos ./autenticos/ --falsos ./falsos/
"""

import os
import argparse
from pathlib import Path

import cv2
import numpy as np


TEMPLATE_SIZE = (100, 100)
HIST_BINS = 32
LAPLACIAN_BINS = 32
FEATURE_DIM = HIST_BINS + LAPLACIAN_BINS

ORB_FEATURES = 2000
FLANN_CHECKS = 100
RANSAC_THRESHOLD = 5.0
MATCH_RATIO = 0.75

RF_ESTIMATORS = 150
RF_MAX_DEPTH = 15


class MultiTemplateAligner:
    def __init__(self, templates_dir):
        self.templates = {}
        self.template_kp = {}
        self.template_des = {}

        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=1.2, nlevels=1, edgeThreshold=3)
        index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
        search_params = dict(checks=FLANN_CHECKS)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        for path in sorted(Path(templates_dir).glob("*.png")):
            ft = path.stem
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, TEMPLATE_SIZE)
            self.templates[ft] = img
            kp, des = self.orb.detectAndCompute(img, None)
            self.template_kp[ft] = kp if kp else []
            self.template_des[ft] = des

        print(f"[ALIGNER] {len(self.templates)} templates: {sorted(self.templates.keys())}")

    def align(self, img, feature_type):
        if feature_type not in self.templates or img is None or img.size == 0:
            return None
        img_kp, img_des = self.orb.detectAndCompute(img, None)
        if img_des is not None and len(img_des) >= 4:
            try:
                td = self.template_des.get(feature_type)
                if td is not None and len(td) >= 4:
                    matches = self.flann.knnMatch(td, img_des, k=2)
                    good = [m for m, n in matches if m.distance < MATCH_RATIO * n.distance]
                    if len(good) >= 4:
                        src_pts = np.float32([self.template_kp[feature_type][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([img_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, RANSAC_THRESHOLD)
                        if H is not None:
                            return cv2.warpPerspective(img, H, TEMPLATE_SIZE, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            except Exception:
                pass
        return cv2.resize(img, TEMPLATE_SIZE)


def extract_features(img):
    hist = cv2.calcHist([img], [0], None, [HIST_BINS], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = np.abs(lap).astype(np.uint8)
    lap_hist = cv2.calcHist([lap], [0], None, [LAPLACIAN_BINS], [0, 256])
    lap_hist = cv2.normalize(lap_hist, lap_hist).flatten()
    return np.concatenate([hist, lap_hist]).astype(np.float32)


def _detect_feature_type(path):
    fname = path.stem
    known = ["valor_", "animal_", "ir_", "personaje_", "serie_a"]
    for prefix in known:
        if fname.startswith(prefix):
            end = fname.find('_', len(prefix))
            if end == -1: end = len(fname)
            return fname[:end]
    for part in path.parts:
        for prefix in known:
            if part.startswith(prefix):
                end = part.find('_', len(prefix))
                if end == -1: end = len(part)
                return part[:end]
    return None


def train(aligner, autenticos_dir, falsos_dir, output_model):
    X, y = [], []

    for label, folder in [(1, autenticos_dir), (0, falsos_dir)]:
        folder = Path(folder)
        paths = sorted(list(folder.rglob("*.png")) + list(folder.rglob("*.jpg")) + list(folder.rglob("*.bmp")))
        name = "autenticas" if label else "falsas"
        print(f"\nProcesando {len(paths)} imagenes {name}...")

        for i, p in enumerate(paths):
            ft = _detect_feature_type(p)
            if ft is None or ft == "serie_a":
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            aligned = aligner.align(img, ft)
            if aligned is None:
                continue
            X.append(extract_features(aligned))
            y.append(label)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(paths)}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"\n[FEATURES] {X.shape[0]} muestras, {X.shape[1]} dims")
    print(f"  Autenticas: {sum(y==1)}, Falsas: {sum(y==0)}")

    # Shuffle y split 80/20
    idx = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(idx)
    split = int(0.8 * len(X))
    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_test, y_test = X[idx[split:]], y[idx[split:]]
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # OpenCV Random Forest
    rf = cv2.ml.RTrees_create()
    rf.setMaxDepth(RF_MAX_DEPTH)
    rf.setMinSampleCount(2)
    rf.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, RF_ESTIMATORS, 0.01))

    train_data = cv2.ml.TrainData_create(X_train, cv2.ml.ROW_SAMPLE, y_train)
    rf.train(train_data)

    # Evaluar
    y_pred = []
    for row in X_test:
        _, result = rf.predict(row.reshape(1, -1))
        y_pred.append(int(result[0]))
    y_pred = np.array(y_pred)

    acc = np.mean(y_pred == y_test)
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    tn = np.sum((y_pred == 0) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n[RESULTADOS]")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  TP:{tp} FP:{fp} TN:{tn} FN:{fn}")

    # Guardar
    rf.save(str(output_model))
    print(f"\n[MODELO] Exportado a {output_model}")

    return rf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", required=True)
    parser.add_argument("--autenticos", required=True)
    parser.add_argument("--falsos", required=True)
    parser.add_argument("--output", default="rf_classifier.xml")
    args = parser.parse_args()

    aligner = MultiTemplateAligner(args.templates)
    train(aligner, args.autenticos, args.falsos, args.output)


if __name__ == "__main__":
    main()
