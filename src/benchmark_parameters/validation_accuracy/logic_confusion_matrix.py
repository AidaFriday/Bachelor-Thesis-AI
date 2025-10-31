import os
import cv2
import numpy as np
from tqdm import tqdm
import sys
import json
from sklearn.metrics import confusion_matrix

import sys, os

# ✅ Correct path: go up 2 directories (to reach src/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from connector import load_model


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def collect_pairs(dataset_path, max_pairs=300):
    """Collect ~50% positive and ~50% negative LFW-style pairs."""
    people = sorted(
        [
            p
            for p in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, p))
        ]
    )

    pos_pairs = []
    neg_pairs = []

    # Positive pairs (same identity, first two images)
    for person in people:
        imgs = sorted(
            [
                os.path.join(dataset_path, person, f)
                for f in os.listdir(os.path.join(dataset_path, person))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        if len(imgs) >= 2:
            pos_pairs.append((imgs[0], imgs[1], 1))

    # Negative pairs (adjacent identity folders)
    for i in range(len(people) - 1):
        p1 = people[i]
        p2 = people[i + 1]

        imgs1 = sorted(
            [
                f
                for f in os.listdir(os.path.join(dataset_path, p1))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        imgs2 = sorted(
            [
                f
                for f in os.listdir(os.path.join(dataset_path, p2))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        if imgs1 and imgs2:
            neg_pairs.append(
                (
                    os.path.join(dataset_path, p1, imgs1[0]),
                    os.path.join(dataset_path, p2, imgs2[0]),
                    0,
                )
            )

    half = max_pairs // 2
    pairs = pos_pairs[:half] + neg_pairs[:half]
    return pairs[:max_pairs]


def compute_confusion_from_scores(scores, labels):
    """Pick best threshold & compute confusion matrix."""
    scores = np.array(scores)
    labels = np.array(labels)

    thresholds = np.unique(scores)
    best_acc = -1
    best_t = 0.5

    for t in thresholds:
        preds = (scores >= t).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = t

    preds = (scores >= best_t).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(best_acc),
        "threshold": float(best_t),
    }


def run_confusion(model_name, dataset_path, iters=300):
    print(f"[CM] Model: {model_name}")
    print(f"[CM] Dataset: {dataset_path}")

    wrapper = load_model(model_name)

    # Support "lfw-deepfunneled" auto-pathing
    if os.path.isdir(os.path.join(dataset_path, "lfw-deepfunneled")):
        dataset_path = os.path.join(dataset_path, "lfw-deepfunneled")

    pairs = collect_pairs(dataset_path, max_pairs=iters)

    sims = []
    labels = []

    for img1, img2, label in tqdm(pairs, desc="Computing Confusion Matrix"):
        a = cv2.imread(img1)
        b = cv2.imread(img2)
        if a is None or b is None:
            continue

        emb1 = wrapper.embed(a)
        emb2 = wrapper.embed(b)
        if emb1 is None or emb2 is None:
            continue

        sims.append(cosine_similarity(emb1, emb2))
        labels.append(label)

    result = compute_confusion_from_scores(sims, labels)

    # GUI-ready JSON output
    print(
        json.dumps(
            {
                "kind": "confusion_matrix",
                "model": model_name,
                "dataset": os.path.basename(dataset_path),
                "pairs_tested": len(labels),
                **result,
            }
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--iters", type=int, default=300)
    args = parser.parse_args()

    run_confusion(args.model, args.dataset, args.iters)
