import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import sys
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime

# make project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from connector import load_model


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _resolve_video_root(dataset_path: str) -> str:
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
    Same deterministic pairing as confusion_matrix_video.
    """

    root = _resolve_video_root(dataset_path)
    return_all = start_identity == "__ALL__" or max_pairs is None or max_pairs == -1

    people = sorted(
        [
            d
            for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ]
    )

    if not return_all and start_identity in people:
        start_idx = people.index(start_identity)
        people = people[start_idx:]

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
            for f in imgs[:max_frames_per_clip]:
                frames.append(os.path.join(clip_dir, f))

        if len(frames) >= 2:
            all_person_frames[p] = frames

    pos_pairs = []
    neg_pairs = []

    for person in people:
        frames = all_person_frames.get(person, [])
        if len(frames) < 2:
            continue

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


def run_roc(model_name, dataset_path, start_identity, iters=300):
    print(f"[ROC-VIDEO] Model: {model_name}")
    print(f"[ROC-VIDEO] Dataset: {dataset_path}")
    print(f"[ROC-VIDEO] Pairs: {iters}")

    wrapper = load_model(model_name)

    root = _resolve_video_root(dataset_path)

    if start_identity == "__ALL__":
        pairs = collect_pairs(root, start_identity="__ALL__", max_pairs=None)
    else:
        pairs = collect_pairs(root, start_identity=start_identity, max_pairs=iters)

    sims = []
    labels = []
    used_identities = {}
    pair_records = []

    total = len(pairs)

    for i, (img1, img2, label) in enumerate(pairs, start=1):
        # progress for GUI
        sys.stdout.write(
            json.dumps({"_type": "progress", "progress": i, "total": total}) + "\n"
        )
        sys.stdout.flush()

        a = cv2.imread(img1)
        b = cv2.imread(img2)
        if a is None or b is None:
            continue

        # detect faces
        faces_a = wrapper.detector.detect(a)
        faces_b = wrapper.detector.detect(b)
        if not faces_a or not faces_b:
            continue

        # align using landmarks
        aligned_a = wrapper.detector.align_for(a, faces_a[0]["kps"])
        aligned_b = wrapper.detector.align_for(b, faces_b[0]["kps"])
        if aligned_a is None or aligned_b is None:
            continue

        emb1 = wrapper.embed(aligned_a)
        emb2 = wrapper.embed(aligned_b)
        if emb1 is None or emb2 is None:
            continue

        sim = cosine_similarity(emb1, emb2)
        sims.append(sim)
        labels.append(label)

        # track identities used
        person1 = os.path.basename(os.path.dirname(os.path.dirname(img1)))
        person2 = os.path.basename(os.path.dirname(os.path.dirname(img2)))
        used_identities.setdefault(person1, set()).add(os.path.basename(img1))
        used_identities.setdefault(person2, set()).add(os.path.basename(img2))

        pair_records.append(
            {
                "img1": img1.replace(root + os.sep, ""),
                "img2": img2.replace(root + os.sep, ""),
                "label": "pos" if label == 1 else "neg",
                "similarity": float(sim),
            }
        )

    sims = np.array(sims)
    labels = np.array(labels)

    if len(labels) == 0:
        print(json.dumps({"error": "No valid pairs evaluated"}))
        return

    # ROC
    fpr, tpr, thresholds = roc_curve(labels, sims)
    roc_auc = auc(fpr, tpr)

    # best threshold (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    print(f"[THRESHOLD-VIDEO] Best threshold (Youden J): {best_threshold:.4f}")

    # EER
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0

    # TAR @ FAR = 1e-3
    target_far = 1e-3
    idx = np.searchsorted(fpr, target_far, side="right") - 1
    if 0 <= idx < len(tpr):
        tar_at_far = tpr[idx]
    else:
        tar_at_far = float("nan")

    export = {
        "meta": {
            "model": model_name,
            "dataset": os.path.basename(root),
            "test_name": "ROC (Video)",
            "pairs_evaluated": len(labels),
        },
        "metrics": {
            "auc": float(roc_auc),
            "eer": float(eer),
            "best_threshold": float(best_threshold),
            "tar_far_1e3": float(tar_at_far),
        },
        "pairs": pair_records,
        "identities_used": {
            p: sorted(list(imgs)) for p, imgs in used_identities.items()
        },
    }

    # save export JSON
    export_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "exports",
        f"{model_name}_roc_video_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    with open(export_path, "w") as f:
        json.dump(export, f, indent=2)

    print(f"[EXPORTED VIDEO ROC] -> {os.path.abspath(export_path)}")

    # save ROC plot PNG
    save_path = os.path.join(os.path.dirname(__file__), "roc_video_result.png")
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Video) – {model_name}")
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(
        json.dumps(
            {
                "kind": "roc_image",
                "path": save_path,
                "auc": float(roc_auc),
                "eer": float(eer),
                "best_threshold": float(best_threshold),
                "tar_far_1e3": float(tar_at_far),
                "pairs_tested": int(len(labels)),
                "model": model_name,
                "dataset": os.path.basename(root),
            }
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--iters", type=int, default=300)
    args = parser.parse_args()
    run_roc(args.model, args.dataset, args.start, args.iters)
