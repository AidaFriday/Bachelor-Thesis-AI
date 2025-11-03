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
FIXED_THRESHOLD = 0.80


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


from itertools import combinations
import random


def collect_pairs(dataset_path, start_identity=None, max_pairs=None):
    """
    Deterministic pair generation:
    - Start from selected identity unless ALL mode is activated
    - Max ~10 positive pairs per identity
    - Balanced negative pairs
    - No randomness
    """

    # ✅ Detect ALL mode
    return_all = start_identity == "__ALL__" or max_pairs is None or max_pairs == -1

    # List all people
    people = sorted(
        [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]
    )

    # ✅ If not ALL mode and identity was selected → reorder dataset starting from that identity
    if not return_all and start_identity in people:
        start_idx = people.index(start_identity)
        people = people[start_idx:]

    # Load images once
    all_people_imgs = {
        p: sorted(
            [
                os.path.join(dataset_path, p, f)
                for f in os.listdir(os.path.join(dataset_path, p))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        for p in people
    }

    pos_pairs = []
    neg_pairs = []

    # ----- Positive pairs (max 10 per identity) -----
    for person in people:
        imgs = all_people_imgs[person]
        if len(imgs) < 2:
            continue

        for i in range(min(len(imgs) - 1, 9)):
            pos_pairs.append((imgs[i], imgs[i + 1], 1))

        # ✅ Only stop if not ALL mode
        if not return_all and len(pos_pairs) >= max_pairs // 2:
            break

    # ✅ Determine how many negative pairs needed
    pos_count = len(pos_pairs)
    neg_needed = pos_count

    # ----- Negative pairs (identity i vs identity i+1) -----
    for i in range(len(people) - 1):
        p1, p2 = people[i], people[i + 1]
        imgs1, imgs2 = all_people_imgs[p1], all_people_imgs[p2]

        for a, b in zip(imgs1, imgs2):
            neg_pairs.append((a, b, 0))
            if not return_all and len(neg_pairs) >= neg_needed:
                break

        if not return_all and len(neg_pairs) >= neg_needed:
            break

    neg_pairs = neg_pairs[:neg_needed]

    final_pairs = pos_pairs + neg_pairs

    # ✅ In ALL mode return everything, else return limited
    if return_all:
        return final_pairs
    else:
        return final_pairs[:max_pairs]


def run_confusion(model_name, dataset_path, start_identity, iters=300):
    print(f"[CM] Model: {model_name}")
    print(f"[CM] Dataset: {dataset_path}")
    print(f"[CM] Using FIXED THRESHOLD = {FIXED_THRESHOLD}")

    wrapper = load_model(model_name)

    if os.path.isdir(os.path.join(dataset_path, "lfw-deepfunneled")):
        dataset_path = os.path.join(dataset_path, "lfw-deepfunneled")

    if start_identity == "__ALL__":
        pairs = collect_pairs(dataset_path, start_identity=None, max_pairs=None)
    else:
        pairs = collect_pairs(dataset_path, start_identity, max_pairs=iters)

    sims = []
    labels = []
    pair_records = []
    used_identities = {}

    total = len(pairs)

    for i, (img1, img2, label) in enumerate(pairs, start=1):

        # ---- PROGRESS UPDATE FOR GUI ----
        sys.stdout.write(
            json.dumps({"_type": "progress", "progress": i, "total": total}) + "\n"
        )
        sys.stdout.flush()

        # ✅ Track which identities and images are used
        person1 = os.path.basename(os.path.dirname(img1))
        person2 = os.path.basename(os.path.dirname(img2))
        used_identities.setdefault(person1, set()).add(os.path.basename(img1))
        used_identities.setdefault(person2, set()).add(os.path.basename(img2))

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

    pos_count = int((labels == 1).sum())
    neg_count = int((labels == 0).sum())

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
        "pos_pairs": pos_count,
        "neg_pairs": neg_count,
    }

    identities_detail = {
        person: sorted(list(images)) for person, images in used_identities.items()
    }

    # ✅ Build export JSON
    export_data = {
        "meta": {
            "model": model_name,
            "dataset": os.path.basename(dataset_path),
            "test_name": "Confusion Matrix (Image)",
            "threshold": FIXED_THRESHOLD,
            "pairs_evaluated": len(labels),
            "pos_pairs": pos_count,
            "neg_pairs": neg_count,
            "start_identity": start_identity,
            "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        },
        "metrics": result,
        "pairs": pair_records,
        "identities_used": identities_detail,
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
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--iters", type=int, default=300)

    args = parser.parse_args()
    run_confusion(args.model, args.dataset, args.start, args.iters)
