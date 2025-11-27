# confusion_custom_pairs_fixed.py
import os

os.environ["MPLBACKEND"] = "Agg"

import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)

from connector import load_model
from models.wrap_facedetection import FaceDetectorAligner


def cosine_similarity(a, b):
    a = np.asarray(a, np.float32).flatten()
    b = np.asarray(b, np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else -1.0


def load_pairs(pairs_file, dataset_root):
    pairs = []
    fold_ids = []
    with open(pairs_file, "r") as f:
        lines = f.read().strip().split("\n")

    num_folds, pairs_per_fold = map(int, lines[0].split())
    idx = 1

    for fold in range(num_folds):
        # positive pairs
        for _ in range(pairs_per_fold):
            p, i1, i2 = lines[idx].split()
            img1 = os.path.join(dataset_root, p, i1)
            img2 = os.path.join(dataset_root, p, i2)
            pairs.append((img1, img2, 1))
            fold_ids.append(fold)
            idx += 1

        # negative pairs
        for _ in range(pairs_per_fold):
            p1, i1, p2, i2 = lines[idx].split()
            img1 = os.path.join(dataset_root, p1, i1)
            img2 = os.path.join(dataset_root, p2, i2)
            pairs.append((img1, img2, 0))
            fold_ids.append(fold)
            idx += 1

    return pairs, np.array(fold_ids, np.int32)


def find_best_threshold(sims, labels):
    thrs = np.unique(sims)
    best_thr, best_acc = 0, -1
    for t in thrs:
        preds = (sims >= t).astype(np.int32)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc, best_thr = acc, t
    return best_thr, best_acc


def plot_confusion_matrix_lfw_style(cm, model_name, out_path):
    """
    EXACTLY matches LFW orientation and labels.
    Rows   = Actual (Negative, Positive)
    Cols   = Predicted (Negative, Positive)
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    # Title
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=18)

    # LFW labels
    ax.set_xlabel("Predicted label", fontsize=16)
    ax.set_ylabel("Actual label", fontsize=16)

    # Tick labels (IMPORTANT: LFW order)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"], fontsize=14)
    ax.set_yticklabels(["Negative", "Positive"], fontsize=14)

    # Numbers inside the boxes
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=15,
            )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_confusion(model_name, dataset_root, pairs_file):
    print(f"[INFO] Running confusion matrix: {model_name}")

    wrapper = load_model(model_name)
    aligner = FaceDetectorAligner(device="cpu")

    pairs, fold_ids = load_pairs(pairs_file, dataset_root)
    sims = []
    labels = []

    # ---- compute similarities ----
    for im1, im2, lab in pairs:
        a = cv2.imread(im1)
        b = cv2.imread(im2)

        if a is None or b is None:
            sims.append(-1 if lab == 1 else 2)
            labels.append(lab)
            continue

        f1 = aligner.detect(a)
        f2 = aligner.detect(b)

        if not f1 or not f2:
            sims.append(-1 if lab == 1 else 2)
            labels.append(lab)
            continue

        fa = aligner.align_for(a, f1[0]["kps"])
        fb = aligner.align_for(b, f2[0]["kps"])

        if fa is None or fb is None:
            sims.append(-1 if lab == 1 else 2)
        else:
            emb1 = wrapper.embed(fa)
            emb2 = wrapper.embed(fb)
            sims.append(cosine_similarity(emb1, emb2))

        labels.append(lab)

    sims = np.array(sims, np.float32)
    labels = np.array(labels, np.int32)

    # ---- Find best global threshold ----
    thr, acc = find_best_threshold(sims, labels)
    preds = (sims >= thr).astype(np.int32)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    print(f"[THRESHOLD] best={thr:.4f}")
    print(f"[ACCURACY] {acc * 100:.2f}%")
    print(f"TN={tn}, FP={fp}")
    print(f"FN={fn}, TP={tp}")

    # ---- Save JSON ----
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_clean = model_name.lower().replace(" ", "_")

    summary = {
        "model": model_name,
        "folds": int(fold_ids.max() + 1),
        "pairs": len(labels),
        "best_threshold": float(thr),
        "accuracy": float(acc),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    json_path = f"{model_clean}_custom_confusion_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Saved JSON: {json_path}")

    # ---- Save LFW-style PNG ----
    # LFW requires: rows = actual, columns = predicted
    cm = np.array([[tn, fp], [fn, tp]])  # Actual Negative  # Actual Positive

    png_path = f"{model_clean}_custom_confusion_{timestamp}.png"
    plot_confusion_matrix_lfw_style(cm, model_name, png_path)

    print(f"[OK] Saved PNG: {png_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--pairs", required=True)
    args = p.parse_args()

    run_confusion(args.model, args.dataset, args.pairs)
