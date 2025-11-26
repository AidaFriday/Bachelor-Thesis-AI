import os
os.environ["MPLBACKEND"] = "Agg"

import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import numpy as np

# FIX: always add project 'src' root to Python import path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)

from models.wrap_yolov5face import YOLOv5FaceDetector
from connector import load_model


# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
YOLOV5_ONNX = os.path.abspath("../../external/FaceNet_onnx/yolov5s-face.onnx")


# ----------------------------------------------------------
# SIMILARITY
# ----------------------------------------------------------
def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ----------------------------------------------------------
# LOAD PAIRS (custom dataset, .jpeg, 3 digits)
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

    return pairs, np.asarray(fold_ids, dtype=np.int32)


# ----------------------------------------------------------
# ALIGNMENT (ArcFace 5-point template)
# ----------------------------------------------------------
class YoloAligner:
    def __init__(self, onnx_path):
        self.detector = YOLOv5FaceDetector(onnx_path)

        # Standard ArcFace landmark template
        self.template = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)

    def align(self, img):
        dets = self.detector.get(img)
        if len(dets) == 0:
            return None

        kps = dets[0]["kps"]
        M = cv2.estimateAffinePartial2D(kps, self.template)[0]
        if M is None:
            return None

        aligned = cv2.warpAffine(img, M, (112, 112))
        return aligned


# ----------------------------------------------------------
# FIND BEST GLOBAL THRESHOLD
# ----------------------------------------------------------
def find_best_threshold(sims, labels):
    unique_thr = np.unique(sims)
    best_acc = -1.0
    best_thr = 0.0

    for t in unique_thr:
        preds = (sims >= t).astype(np.int32)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc = acc
            best_thr = t

    return float(best_thr), float(best_acc)


# ----------------------------------------------------------
# CONFUSION MATRIX PROTOCOL
# ----------------------------------------------------------
def run_custom_confusion(model_name, dataset_path, pairs_file, threshold=None):

    start = datetime.now()

    print("[CONF] Loading model:", model_name)
    wrapper = load_model(model_name)

    aligner = YoloAligner(YOLOV5_ONNX)
    pairs, fold_ids = load_pairs(pairs_file, dataset_path)

    sims = []
    labels = []

    total = len(pairs)
    print(f"[CONF] Total pairs: {total}")

    for i, (img1, img2, label) in enumerate(pairs, start=1):
        if i % 20 == 0 or i == 1 or i == total:
            print(f"[CONF] {i}/{total}")

        a = cv2.imread(img1)
        b = cv2.imread(img2)

        emb1 = emb2 = None
        error = False

        if a is None or b is None:
            error = True
        else:
            face1 = aligner.align(a)
            face2 = aligner.align(b)
            if face1 is None or face2 is None:
                error = True
            else:
                emb1 = wrapper.embed(face1)
                emb2 = wrapper.embed(face2)

        if error:
            # force misclassification so the pair contributes correctly
            sim = -1.0 if label == 1 else 2.0
        else:
            sim = cosine_similarity(emb1, emb2)

        sims.append(sim)
        labels.append(label)

    sims = np.asarray(sims, np.float32)
    labels = np.asarray(labels, np.int32)

    # ------------------------------------------------------
    # Choose threshold
    # ------------------------------------------------------
    if threshold is None:
        best_thr, best_acc = find_best_threshold(sims, labels)
        print(f"[CONF] Best threshold (global): {best_thr:.4f}")
        print(f"[CONF] Accuracy at best threshold: {best_acc * 100:.2f}%")
    else:
        best_thr = float(threshold)
        preds_tmp = (sims >= best_thr).astype(np.int32)
        best_acc = float(np.mean(preds_tmp == labels))
        print(f"[CONF] Using user threshold: {best_thr:.4f}")
        print(f"[CONF] Accuracy: {best_acc * 100:.2f}%")

    # ------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------
    preds = (sims >= best_thr).astype(np.int32)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    print("\n[CONF] Confusion Matrix")
    print("        Pred 0    Pred 1")
    print(f"Act 0    {tn:4d}      {fp:4d}")
    print(f"Act 1    {fn:4d}      {tp:4d}\n")

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    tnr = tn / (tn + fp) if tn + fp else 0
    fpr = 1 - tnr
    fnr = 1 - recall

    end = datetime.now()

    # ------------------------------------------------------
    # Plot confusion matrix
    # ------------------------------------------------------
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int32)

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.title(f"Confusion Matrix – {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    out_png = "custom_confusion.png"
    plt.savefig(out_png, dpi=150)
    print(f"[CONF] Saved PNG: {out_png}")

    # ------------------------------------------------------
    # JSON summary
    # ------------------------------------------------------
    result = {
        "kind": "custom_confusion_matrix",
        "model": model_name,
        "dataset": os.path.basename(dataset_path),
        "pairs": int(len(labels)),
        "threshold": float(best_thr),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "tpr": float(recall),
        "tnr": float(tnr),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "elapsed_sec": float((end - start).total_seconds()),
    }

    out_json = "custom_confusion.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[CONF] Saved JSON: {out_json}")

    return result


# ----------------------------------------------------------
# CLI ENTRY POINT
# ----------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pairs", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    run_custom_confusion(
        args.model,
        args.dataset,
        args.pairs,
        threshold=args.threshold,
    )
