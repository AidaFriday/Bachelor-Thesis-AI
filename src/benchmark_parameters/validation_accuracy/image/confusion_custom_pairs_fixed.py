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
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else -1.0


def load_pairs(pairs_file, dataset_root):
    pairs = []
    fold_ids = []
    with open(pairs_file, "r") as f:
        lines = f.read().strip().split("\n")

    num_folds, pairs_per_fold = map(int, lines[0].split())
    idx = 1

    for fold in range(num_folds):
        # positive
        for _ in range(pairs_per_fold):
            p, i1, i2 = lines[idx].split()
            img1 = os.path.join(dataset_root, p, i1)
            img2 = os.path.join(dataset_root, p, i2)
            pairs.append((img1, img2, 1))
            fold_ids.append(fold)
            idx += 1

        # negative
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
    best_t, best_a = 0, -1
    for t in thrs:
        preds = (sims >= t).astype(np.int32)
        acc = np.mean(preds == labels)
        if acc > best_a:
            best_a, best_t = acc, t
    return best_t, best_a


def run_confusion(model_name, dataset_root, pairs_file):
    print(f"[INFO] Running confusion matrix: {model_name}")

    wrapper = load_model(model_name)
    aligner = FaceDetectorAligner(device="cpu")

    pairs, fold_ids = load_pairs(pairs_file, dataset_root)
    sims = []
    labels = []

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

    # ---- Best global threshold ----
    thr, acc = find_best_threshold(sims, labels)
    print(f"[BEST THRESHOLD] {thr:.4f}")
    print(f"[ACCURACY] {acc*100:.2f}%")

    preds = (sims >= thr).astype(np.int32)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    print("Confusion Matrix:")
    print(f"TN={tn} FP={fp}")
    print(f"FN={fn} TP={tp}")

    # ---------- SAVE JSON + PNG ----------
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_clean = model_name.lower().replace(" ", "_")

    # JSON
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

    json_name = f"{model_clean}_custom_confusion_{timestamp}.json"
    with open(json_name, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Saved JSON: {json_name}")

    # PNG confusion matrix
    png_name = f"{model_clean}_custom_confusion_{timestamp}.png"

    plt.figure(figsize=(5, 4))
    cm = np.array([[tn, fp], [fn, tp]])
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()

    for (x, y), value in np.ndenumerate(cm):
        plt.text(y, x, str(value), ha="center", va="center", color="black", fontsize=12)

    plt.xticks([0, 1], ["Actual 0", "Actual 1"])
    plt.yticks([0, 1], ["Pred 0", "Pred 1"])
    plt.tight_layout()
    plt.savefig(png_name, dpi=150)
    plt.close()

    print(f"[OK] Saved PNG: {png_name}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--pairs", required=True)
    args = p.parse_args()

    run_confusion(args.model, args.dataset, args.pairs)
