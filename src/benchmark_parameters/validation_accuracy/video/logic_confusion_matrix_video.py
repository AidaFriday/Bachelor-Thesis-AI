# Builds deterministic positive/negative image pairs from video folders,
# runs face detection + alignment + embedding, computes confusion metrics,
# and exports detailed per-pair JSON results.

import os
import cv2
import numpy as np
from tqdm import tqdm
import sys
import json
from datetime import datetime

# go up 3 levels to project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from connector import load_model


# <<< HARD-CODED THRESHOLD >>>
FIXED_THRESHOLD = 0.60


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _resolve_video_root(dataset_path: str) -> str:
    """
    YTF-style: dataset may be either:
      - <root>/aligned_images_DB/person/clip/frame.jpg
      - or directly at aligned_images_DB
    This normalizes to the level that contains the person folders.
    """
    if os.path.isdir(os.path.join(dataset_path, "aligned_images_DB")):
        return os.path.join(dataset_path, "aligned_images_DB")
    return dataset_path


def collect_pairs(
    dataset_path,
    start_identity=None,
    max_pairs=None,
    max_frames_per_clip=5,
    max_pos_per_identity=10,
    max_neg_per_identity=20,
):
    """
    Deterministic pair generation for VIDEO datasets.

    - Walks: person / clip / frame.jpg
    - Per clip: up to `max_frames_per_clip` frames (first N, sorted)
    - Positive pairs:
        * up to `max_pos_per_identity` adjacent-frame pairs per identity
    - Negative pairs:
        * person[i] frames vs person[i+1] frames (same index), up to
          `max_neg_per_identity` per adjacent-identity pair
    - No randomness, so results are reproducible across runs/models.

    start_identity:
        "__ALL__"  → use all identities
        some name → start from that person (top-down), then go downwards
    """

    root = _resolve_video_root(dataset_path)

    # --- detect ALL mode ---
    return_all = start_identity == "__ALL__" or max_pairs is None or max_pairs == -1

    # --- list identities ---
    people = sorted(
        [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    )

    # reorder so we start from a specific identity
    if not return_all and start_identity in people:
        start_idx = people.index(start_identity)
        people = people[start_idx:]

    # --- collect frames per identity (limited) ---
    all_person_frames = {}
    for p in people:
        person_dir = os.path.join(root, p)
        clips = sorted(
            [
                c
                for c in os.listdir(person_dir)
                if os.path.isdir(os.path.join(person_dir, c))
            ]
        )

        frames = []
        for c in clips:
            clip_dir = os.path.join(person_dir, c)
            imgs = sorted(
                f
                for f in os.listdir(clip_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            # take first N frames from each clip for determinism & speed
            for f in imgs[:max_frames_per_clip]:
                frames.append(os.path.join(clip_dir, f))

        if len(frames) >= 2:
            all_person_frames[p] = frames

    pos_pairs = []
    neg_pairs = []

    # --- positive pairs (adjacent frames) ---
    for person in people:
        frames = all_person_frames.get(person, [])
        if len(frames) < 2:
            continue

        # adjacency pairs: (f0,f1), (f1,f2), ...
        limit = min(len(frames) - 1, max_pos_per_identity)
        for i in range(limit):
            pos_pairs.append((frames[i], frames[i + 1], 1))

        if not return_all and max_pairs is not None:
            if len(pos_pairs) >= max_pairs // 2:
                break

    pos_count = len(pos_pairs)
    if pos_count == 0:
        return []

    neg_needed = pos_count

    # --- negative pairs (person i vs person i+1) ---
    for i in range(len(people) - 1):
        p1, p2 = people[i], people[i + 1]
        f1 = all_person_frames.get(p1, [])
        f2 = all_person_frames.get(p2, [])
        if not f1 or not f2:
            continue

        limit = min(len(f1), len(f2), max_neg_per_identity)
        for j in range(limit):
            neg_pairs.append((f1[j], f2[j], 0))
            if not return_all and len(neg_pairs) >= neg_needed:
                break

        if not return_all and len(neg_pairs) >= neg_needed:
            break

    neg_pairs = neg_pairs[:neg_needed]
    all_pairs = pos_pairs + neg_pairs

    if return_all:
        return all_pairs
    else:
        return all_pairs[:max_pairs]


def run_confusion(model_name, dataset_path, start_identity, iters=300):
    print(f"[CM-VIDEO] Model: {model_name}")
    print(f"[CM-VIDEO] Dataset: {dataset_path}")
    print(f"[CM-VIDEO] Using FIXED THRESHOLD = {FIXED_THRESHOLD}")

    wrapper = load_model(model_name)

    # normalize root + build pairs
    root = _resolve_video_root(dataset_path)

    if start_identity == "__ALL__":
        pairs = collect_pairs(root, start_identity="__ALL__", max_pairs=None)
    else:
        pairs = collect_pairs(root, start_identity=start_identity, max_pairs=iters)

    sims = []
    labels = []
    pair_records = []
    used_identities = {}

    total = len(pairs)
    failed_pairs = 0

    for i, (img1, img2, label) in enumerate(pairs, start=1):
        # progress for GUI
        sys.stdout.write(
            json.dumps({"_type": "progress", "progress": i, "total": total}) + "\n"
        )
        sys.stdout.flush()

        error = False

        a = cv2.imread(img1)
        b = cv2.imread(img2)

        error = False
        emb1 = emb2 = None

        if a is None or b is None:
            error = True
        else:
            # --- DETECT ---
            faces_a = wrapper.detector.detect(a)
            faces_b = wrapper.detector.detect(b)

            if not faces_a or not faces_b:
                error = True
            else:
                # --- ALIGN ---
                aligned_a = wrapper.detector.align_for(a, faces_a[0]["kps"])
                aligned_b = wrapper.detector.align_for(b, faces_b[0]["kps"])
                if aligned_a is None or aligned_b is None:
                    error = True
                else:
                    # --- EMBED ---
                    emb1 = wrapper.embed(aligned_a)
                    emb2 = wrapper.embed(aligned_b)
                    if emb1 is None or emb2 is None:
                        error = True

        if error:
            failed_pairs += 1
            # worst-case similarity so it’s always misclassified at any threshold in [0,1]
            if label == 1:
                sim = -1.0  # positive pair → below threshold → FN
            else:
                sim = 2.0  # negative pair → above threshold → FP
        else:
            sim = cosine_similarity(emb1, emb2)

        sims.append(sim)
        labels.append(label)

        # track which identities/images were used (identity = top-level folder)
        person1 = (
            os.path.basename(os.path.dirname(os.path.dirname(img1)))
            if "aligned_images_DB" in img1
            else os.path.basename(os.path.dirname(os.path.dirname(img1)))
        )
        person2 = (
            os.path.basename(os.path.dirname(os.path.dirname(img2)))
            if "aligned_images_DB" in img2
            else os.path.basename(os.path.dirname(os.path.dirname(img2)))
        )

        used_identities.setdefault(person1, set()).add(os.path.basename(img1))
        used_identities.setdefault(person2, set()).add(os.path.basename(img2))

        pair_records.append(
            {
                # store paths relative to ROOT (so reproducible)
                "img1": img1.replace(root + os.sep, ""),
                "img2": img2.replace(root + os.sep, ""),
                "label": "pos" if label == 1 else "neg",
                "similarity": float(sim),
                "error": bool(error),
            }
        )

        scores = np.array(sims)
    labels = np.array(labels)

    if len(labels) == 0:
        print(json.dumps({"error": "No valid pairs evaluated"}))
        return

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
        "dataset": os.path.basename(root),
        "pairs_tested": int(len(labels)),
        "pairs_built": int(total),
        "failed_pairs": int(failed_pairs),
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

    export_data = {
        "meta": {
            "model": model_name,
            "dataset": os.path.basename(root),
            "test_name": "Confusion Matrix (Video)",
            "threshold": FIXED_THRESHOLD,
            "pairs_evaluated": int(len(labels)),
            "pairs_built": int(total),
            "failed_pairs": int(failed_pairs),
            "pos_pairs": pos_count,
            "neg_pairs": neg_count,
            "start_identity": start_identity,
            "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        },
        "metrics": result,
        "pairs": pair_records,
        "identities_used": identities_detail,
    }

    export_dir = os.path.join(os.path.dirname(__file__), "..", "..", "exports")
    os.makedirs(export_dir, exist_ok=True)

    filename = f"{model_name}_confusion_video_{export_data['meta']['timestamp']}.json"
    filepath = os.path.join(export_dir, filename)

    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=4)

    print(f"\n[EXPORTED VIDEO CM] -> {filepath}\n")
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
