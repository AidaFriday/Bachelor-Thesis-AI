import os

os.environ["MPLBACKEND"] = "Agg"

import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt

import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc

# FIX: always add project 'src' root to Python import path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)

from models.wrap_yolov5face import YOLOv5FaceDetector
from connector import load_model


# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
YOLOV5_ONNX = (
    r"C:/programming/Bachelor-Thesis-AI/external/FaceNet_onnx/yolov5s-face.onnx"
)


print("[DEBUG] YOLO model path:", YOLOV5_ONNX)
print("[DEBUG] File exists:", os.path.exists(YOLOV5_ONNX))


# ----------------------------------------------------------
# SAFE SIMILARITY
# ----------------------------------------------------------
def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0 or not np.isfinite(denom):
        return -1.0  # safe fallback

    sim = float(np.dot(a, b) / denom)
    if not np.isfinite(sim):
        return -1.0

    return sim


# ----------------------------------------------------------
# LOAD PAIRS
# ----------------------------------------------------------
def load_pairs(pairs_file, dataset_path):
    pairs = []
    fold_ids = []

    with open(pairs_file, "r") as f:
        lines = f.read().strip().split("\n")

    num_folds, pairs_per_fold = map(int, lines[0].split())
    idx = 1

    def make_name(name, idx):
        return f"{name}_{idx:03d}.jpeg"

    for fold in range(num_folds):
        # positive pairs
        for _ in range(pairs_per_fold):
            name, n1, n2 = lines[idx].split()
            img1 = os.path.join(dataset_path, name, make_name(name, int(n1)))
            img2 = os.path.join(dataset_path, name, make_name(name, int(n2)))
            pairs.append((img1, img2, 1))
            fold_ids.append(fold)
            idx += 1

        # negative pairs
        for _ in range(pairs_per_fold):
            p1, n1, p2, n2 = lines[idx].split()
            img1 = os.path.join(dataset_path, p1, make_name(p1, int(n1)))
            img2 = os.path.join(dataset_path, p2, make_name(p2, int(n2)))
            pairs.append((img1, img2, 0))
            fold_ids.append(fold)
            idx += 1

    return pairs, np.array(fold_ids, dtype=np.int32)


# ----------------------------------------------------------
# ALIGNMENT WRAPPER
# ----------------------------------------------------------
class YoloAligner:
    def __init__(self, onnx_path):
        self.detector = YOLOv5FaceDetector(onnx_path)

        self.template = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )

    def align(self, img):
        dets = self.detector.get(img)
        if len(dets) == 0:
            return None
        kps = dets[0]["kps"]
        M = cv2.estimateAffinePartial2D(kps, self.template)[0]
        if M is None:
            return None
        return cv2.warpAffine(img, M, (112, 112))


# ----------------------------------------------------------
# MAIN ROC PROTOCOL
# ----------------------------------------------------------
def run_custom_roc(model_name, dataset_path, pairs_file):
    print("[INFO] Loading model:", model_name)
    wrapper = load_model(model_name)

    aligner = YoloAligner(YOLOV5_ONNX)

    pairs, fold_ids = load_pairs(pairs_file, dataset_path)

    sims = []
    labels = []

    for i, (img1, img2, label) in enumerate(pairs, start=1):
        a = cv2.imread(img1)
        b = cv2.imread(img2)
        emb1 = emb2 = None
        error = False

        # ---------------------------------------------------
        # DEBUG 1: IMREAD FAILURES
        # ---------------------------------------------------
        if a is None:
            print(f"[ERROR][IMREAD] Could not read image1: {img1}")
            error = True

        if b is None:
            print(f"[ERROR][IMREAD] Could not read image2: {img2}")
            error = True

        if not error:
            # ---------------------------------------------------
            # FACE DETECTION / ALIGNMENT
            # ---------------------------------------------------
            face1 = aligner.align(a)
            face2 = aligner.align(b)

            if face1 is None:
                print(f"[DETECT FAIL] No face detected in image1: {img1}")
                error = True

            if face2 is None:
                print(f"[DETECT FAIL] No face detected in image2: {img2}")
                error = True

            if not error:
                emb1 = wrapper.embed(face1)
                emb2 = wrapper.embed(face2)

                if emb1 is None or emb2 is None:
                    print(f"[EMBED ERROR] Embedding failed for:\n  {img1}\n  {img2}")
                    error = True

        # ---------------------------------------------------
        # SIMILARITY (SAFE)
        # ---------------------------------------------------
        if error:
            sim = -1.0 if label == 1 else 2.0
        else:
            sim = cosine_similarity(emb1, emb2)

            if not np.isfinite(sim):
                print(f"[NAN] Similarity is NaN for:\n  {img1}\n  {img2}")
                sim = -1.0 if label == 1 else 2.0

        sims.append(sim)
        labels.append(label)

    sims = np.asarray(sims, np.float32)
    labels = np.asarray(labels, np.int32)

    # ROC PLOT
    fpr, tpr, thr = roc_curve(labels, sims)
    auc_val = auc(fpr, tpr)

    print("ROC AUC:", auc_val)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
    plt.title("Custom Dataset ROC")

    out = "custom_roc.png"
    plt.savefig(out)
    print("[OK] Saved:", out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pairs", required=True)
    args = parser.parse_args()

    run_custom_roc(args.model, args.dataset, args.pairs)
