# roc_custom_pairs_fixed.py
import os

os.environ["MPLBACKEND"] = "Agg"

import sys
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from datetime import datetime

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


# -----------------------------
# Load pairs with folds (3-fold)
# -----------------------------
def load_pairs_with_folds(pairs_file, dataset_root):
    pairs = []
    fold_ids = []

    with open(pairs_file) as f:
        lines = f.read().strip().split("\n")

    num_folds, pairs_per_fold = map(int, lines[0].split())
    idx = 1

    for fold in range(num_folds):

        # POSITIVE
        for _ in range(pairs_per_fold):
            person, img1, img2 = lines[idx].split()
            img1_path = os.path.join(dataset_root, person, img1)
            img2_path = os.path.join(dataset_root, person, img2)
            pairs.append((img1_path, img2_path, 1))
            fold_ids.append(fold)
            idx += 1

        # NEGATIVE
        for _ in range(pairs_per_fold):
            p1, img1, p2, img2 = lines[idx].split()
            img1_path = os.path.join(dataset_root, p1, img1)
            img2_path = os.path.join(dataset_root, p2, img2)
            pairs.append((img1_path, img2_path, 0))
            fold_ids.append(fold)
            idx += 1

    return pairs, np.array(fold_ids, dtype=np.int32)


# -----------------------------
# Compute per-fold accuracy
# -----------------------------
def compute_fold_accuracy(sims, labels, fold_ids):
    num_folds = int(fold_ids.max()) + 1
    thresholds = []
    accuracies = []

    for k in range(num_folds):
        train_mask = fold_ids != k
        test_mask = fold_ids == k

        sims_train = sims[train_mask]
        labels_train = labels[train_mask]
        sims_test = sims[test_mask]
        labels_test = labels[test_mask]

        unique_thr = np.unique(sims_train)

        best_thr = None
        best_acc = -1

        for t in unique_thr:
            preds = (sims_train >= t).astype(np.int32)
            acc = np.mean(preds == labels_train)
            if acc > best_acc:
                best_acc = acc
                best_thr = t

        preds_test = (sims_test >= best_thr).astype(np.int32)
        test_acc = np.mean(preds_test == labels_test)

        print(f"[Fold {k+1}] thr={best_thr:.4f}, acc={test_acc*100:.2f}%")

        thresholds.append(float(best_thr))
        accuracies.append(float(test_acc))

    thresholds = np.array(thresholds, np.float32)
    accuracies = np.array(accuracies, np.float32)
    return thresholds, accuracies, float(accuracies.mean()), float(accuracies.std())


# -----------------------------
# MAIN
# -----------------------------
def run_custom_roc(model_name, dataset_root, pairs_file):
    print(f"[INFO] Evaluate model: {model_name}")

    wrapper = load_model(model_name)
    aligner = FaceDetectorAligner(device="cpu")

    # Load pairs
    pairs, fold_ids = load_pairs_with_folds(pairs_file, dataset_root)

    sims = []
    labels = []
    valid_fold_ids = []

    # ---------- PROCESS ALL PAIRS ----------
    for (img1, img2, lab), fold in zip(pairs, fold_ids):

        a = cv2.imread(img1)
        b = cv2.imread(img2)
        if a is None or b is None:
            continue

        faces1 = aligner.detect(a)
        faces2 = aligner.detect(b)
        if not faces1 or not faces2:
            continue

        fa = aligner.align_for(a, faces1[0]["kps"])
        fb = aligner.align_for(b, faces2[0]["kps"])
        if fa is None or fb is None:
            continue

        emb1 = wrapper.embed(fa)
        emb2 = wrapper.embed(fb)
        sim = cosine_similarity(emb1, emb2)

        sims.append(sim)
        labels.append(lab)
        valid_fold_ids.append(fold)

    sims = np.asarray(sims, np.float32)
    labels = np.asarray(labels, np.int32)
    fold_ids = np.asarray(valid_fold_ids, np.int32)

    # ---------- 3-FOLD ACC ----------
    thresholds, accs, mean_acc, std_acc = compute_fold_accuracy(sims, labels, fold_ids)

    print(f"\n3-fold accuracy: {mean_acc*100:.4f}% ± {std_acc*100:.4f}%")

    # ---------- GLOBAL ROC ----------
    fpr, tpr, thr = roc_curve(labels, sims)
    auc_val = auc(fpr, tpr)

    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = 0.5 * (fpr[idx] + fnr[idx])

    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thr[best_idx])

    print(f"AUC: {auc_val:.4f}")
    print(f"EER: {eer*100:.2f}%")
    print(f"Best threshold: {best_threshold}")

    # ---------- PLOT ----------
    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC={auc_val:.4f})",
        linewidth=2.2,
        color="#1f77b4",
    )

    plt.plot([0, 1], [0, 1], linestyle="--", color="#ff8c00", linewidth=1.8)

    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title(f"ROC Curve - {model_name} on Custom Dataset", fontsize=15)
    plt.grid(color="#cccccc", linestyle="--", linewidth=0.7)
    plt.legend(loc="lower right", fontsize=12)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.tight_layout()

    # ---------- SAVE PNGs ----------
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_clean = model_name.lower().replace(" ", "_")

    main_png = "custom_roc.png"
    time_png = f"{model_clean}_custom_roc_{timestamp}.png"

    plt.savefig(main_png, dpi=150, bbox_inches="tight")
    plt.savefig(time_png, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved PNGs: {main_png}, {time_png}")

    # ---------- SAVE JSON ----------
    summary = {
        "folds": int(fold_ids.max() + 1),
        "pairs": len(labels),
        "per_fold_accuracy": accs.tolist(),
        "per_fold_threshold": thresholds.tolist(),
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "auc": float(auc_val),
        "eer": float(eer),
        "best_threshold": best_threshold,
    }

    json_filename = f"{model_clean}_custom_roc_{timestamp}.json"
    with open(json_filename, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Saved JSON: {json_filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pairs", required=True)
    args = parser.parse_args()

    run_custom_roc(args.model, args.dataset, args.pairs)
