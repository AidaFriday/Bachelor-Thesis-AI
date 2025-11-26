# roc_custom_pairs_fixed.py
import os

os.environ["MPLBACKEND"] = "Agg"

import sys
import cv2
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)

from connector import load_model
from models.wrap_facedetection import FaceDetectorAligner


def cosine_similarity(a, b):
    a = np.asarray(a, np.float32).flatten()
    b = np.asarray(b, np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else -1.0


def load_pairs_simple(pairs_file, dataset_root):
    pairs = []
    with open(pairs_file) as f:
        for line in f:
            parts = line.strip().split()

            # POSITIVE (3 columns)
            if len(parts) == 3:
                person, img1, img2 = parts
                path1 = os.path.join(dataset_root, person, img1)
                path2 = os.path.join(dataset_root, person, img2)
                pairs.append((path1, path2, 1))

            # NEGATIVE (4 columns)
            elif len(parts) == 4:
                p1, img1, p2, img2 = parts
                path1 = os.path.join(dataset_root, p1, img1)
                path2 = os.path.join(dataset_root, p2, img2)
                pairs.append((path1, path2, 0))

    return pairs


def run_custom_roc(model_name, dataset_root, pairs_file):
    print(f"[INFO] Evaluate model: {model_name}")

    wrapper = load_model(model_name)
    aligner = FaceDetectorAligner(device="cpu")

    # FIXED: simple loader, no folds
    pairs = load_pairs_simple(pairs_file, dataset_root)

    sims = []
    labels = []

    for i, (img1, img2, lab) in enumerate(pairs, 1):

        a = cv2.imread(img1)
        b = cv2.imread(img2)

        if a is None or b is None:
            sim = -1.0 if lab == 1 else 2.0
            sims.append(sim)
            labels.append(lab)
            continue

        faces1 = aligner.detect(a)
        faces2 = aligner.detect(b)

        if not faces1 or not faces2:
            sim = -1.0 if lab == 1 else 2.0
        else:
            fa = aligner.align_for(a, faces1[0]["kps"])
            fb = aligner.align_for(b, faces2[0]["kps"])

            if fa is None or fb is None:
                sim = -1.0 if lab == 1 else 2.0
            else:
                emb1 = wrapper.embed(fa)
                emb2 = wrapper.embed(fb)
                sim = cosine_similarity(emb1, emb2)

        sims.append(sim)
        labels.append(lab)

    sims = np.asarray(sims, np.float32)
    labels = np.asarray(labels, np.int32)

    fpr, tpr, thr = roc_curve(labels, sims)
    auc_val = auc(fpr, tpr)

    print(f"ROC AUC: {auc_val:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} AUC={auc_val:.4f}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve – Custom Dataset")
    plt.grid()
    plt.legend()
    plt.savefig("custom_roc.png", dpi=150)
    print("[OK] Saved ROC image: custom_roc.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pairs", required=True)
    args = parser.parse_args()

    run_custom_roc(args.model, args.dataset, args.pairs)
