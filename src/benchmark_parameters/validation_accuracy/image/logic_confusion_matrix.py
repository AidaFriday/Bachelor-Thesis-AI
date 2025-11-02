import os
import cv2
import numpy as np
from tqdm import tqdm
import sys
import json
from datetime import datetime
from itertools import combinations
import random


# ✅ Correct path: go up 3 directories (image → validation_accuracy → benchmark_parameters → src)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from connector import load_model


# ✅ <<< HARD-CODE YOUR THRESHOLD HERE >>>
FIXED_THRESHOLD = 0.60


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


from itertools import combinations
import random


def collect_pairs(dataset_path, max_pairs=300):
    people = sorted(
        [
            p
            for p in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, p))
        ]
    )

    pos_pairs = []
    neg_pairs = []

    # ----- Build Positive Pairs -----
    for person in people:
        imgs = sorted(
            [
                os.path.join(dataset_path, person, f)
                for f in os.listdir(os.path.join(dataset_path, person))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        # create ALL unique pairs: (img_i, img_j)
        if len(imgs) >= 2:
            for a, b in combinations(imgs, 2):
                pos_pairs.append((a, b, 1))

    # ----- Build Negative Pairs -----
    # Randomly pair images from different identities
    all_people_imgs = {
        p: [
            os.path.join(dataset_path, p, f)
            for f in os.listdir(os.path.join(dataset_path, p))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        for p in people
    }

    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            p1, p2 = people[i], people[j]
            for img1 in all_people_imgs[p1]:
                for img2 in all_people_imgs[p2]:
                    neg_pairs.append((img1, img2, 0))

    # Shuffle and take equal number of pos/neg
    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)

    half = max_pairs // 2
    return pos_pairs[:half] + neg_pairs[:half]


def run_confusion(model_name, dataset_path, iters=300):
    print(f"[CM] Model: {model_name}")
    print(f"[CM] Dataset: {dataset_path}")
    print(f"[CM] Using FIXED THRESHOLD = {FIXED_THRESHOLD}")

    wrapper = load_model(model_name)

    if os.path.isdir(os.path.join(dataset_path, "lfw-deepfunneled")):
        dataset_path = os.path.join(dataset_path, "lfw-deepfunneled")

    pairs = collect_pairs(dataset_path, max_pairs=iters)

    sims = []
    labels = []
    pair_records = []

    for img1, img2, label in tqdm(pairs, desc="Computing Confusion Matrix"):
        a = cv2.imread(img1)
        b = cv2.imread(img2)
        if a is None or b is None:
            continue

        emb1 = wrapper.embed(a)
        emb2 = wrapper.embed(b)
        if emb1 is None or emb2 is None:
            continue

        sim = cosine_similarity(emb1, emb2)
        sims.append(sim)
        labels.append(label)

        # ✅ Store pair data
        pair_records.append(
            {
                "img1": img1.replace(dataset_path + os.sep, ""),
                "img2": img2.replace(dataset_path + os.sep, ""),
                "label": "pos" if label == 1 else "neg",
                "similarity": float(sim),
            }
        )

    scores = np.array(sims)
    labels = np.array(labels)

    preds = (scores >= FIXED_THRESHOLD).astype(int)
    for i in range(len(pair_records)):
        pair_records[i]["predicted"] = int(preds[i])

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)  # TAR
    specificity = tn / (tn + fp + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    far = fp / (fp + tn + 1e-9)  # False Accept Rate
    frr = fn / (fn + tp + 1e-9)  # False Reject Rate

    result = {
        "kind": "confusion_matrix",
        "model": model_name,
        "dataset": os.path.basename(dataset_path),
        "pairs_tested": len(labels),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "far": float(far),
        "frr": float(frr),
        "threshold": float(FIXED_THRESHOLD),
    }

    # ✅ Build export JSON
    export_data = {
        "meta": {
            "model": model_name,
            "dataset": os.path.basename(dataset_path),
            "test_name": "Confusion Matrix (Image)",
            "threshold": FIXED_THRESHOLD,
            "pairs_evaluated": len(labels),
            "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        },
        "metrics": result,
        "pairs": pair_records,
    }

    # ✅ Save to /exports/ folder
    export_dir = os.path.join(os.path.dirname(__file__), "..", "..", "exports")
    os.makedirs(export_dir, exist_ok=True)

    filename = f"{model_name}_confusion_{export_data['meta']['timestamp']}.json"
    filepath = os.path.join(export_dir, filename)

    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=4)

    print(f"\n[EXPORTED] -> {filepath}\n")

    print(json.dumps(result))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--iters", type=int, default=300)

    args = parser.parse_args()
    run_confusion(args.model, args.dataset, args.iters)
