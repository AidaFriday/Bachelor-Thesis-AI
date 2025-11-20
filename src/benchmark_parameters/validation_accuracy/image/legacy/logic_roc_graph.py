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


def dataset_needs_alignment(dataset_path):
    """
    Return False for aligned datasets (e.g., LFW-deepfunneled),
    True for raw datasets.
    """
    path = dataset_path.lower()
    if "lfw" in path or "deepfunneled" in path:
        return False
    return True


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


def run_roc(model_name, dataset_path, start_identity, iters=300):
    print(f"[ROC] Model: {model_name}")
    print(f"[ROC] Dataset: {dataset_path}")
    print(f"[ROC] Pairs: {iters}")

    wrapper = load_model(model_name)

    # ✅ detect if this is ArcFace
    is_arcface = getattr(wrapper, "name", "").lower() == "arcface"

    # ensure correct LFW path
    if os.path.isdir(os.path.join(dataset_path, "lfw-deepfunneled")):
        dataset_path = os.path.join(dataset_path, "lfw-deepfunneled")

    if start_identity == "__ALL__":  # all identities
        pairs = collect_pairs(dataset_path, start_identity=None, max_pairs=None)
    else:  # limited number of pairs starting from identity
        pairs = collect_pairs(dataset_path, start_identity, max_pairs=iters)

    sims = []
    labels = []

    total = len(pairs)
    for i, (img1, img2, label) in enumerate(pairs, start=1):

        # ---- PROGRESS UPDATE (for GUI progress bar) ----
        sys.stdout.write(
            json.dumps({"_type": "progress", "progress": i, "total": total}) + "\n"
        )
        sys.stdout.flush()

        a = cv2.imread(img1)
        b = cv2.imread(img2)

        error = False
        emb1 = emb2 = None

        if a is None or b is None:
            error = True
        else:
            if is_arcface:
                # ✅ ArcFace: use the tested pipeline (detection + alignment inside)
                try:
                    emb1 = wrapper.get_embedding(img1)
                    emb2 = wrapper.get_embedding(img2)
                    if emb1 is None or emb2 is None:
                        error = True
                except Exception:
                    error = True
            else:
                # ✅ Facenet / AdaFace: keep your existing logic
                USE_DETECTION = dataset_needs_alignment(dataset_path)

                if USE_DETECTION:
                    # ---- DETECT ----
                    faces_a = wrapper.detector.detect(a)
                    faces_b = wrapper.detector.detect(b)
                    if not faces_a or not faces_b:
                        error = True
                    else:
                        # ---- ALIGN ----
                        aligned_a = wrapper.detector.align_for(a, faces_a[0]["kps"])
                        aligned_b = wrapper.detector.align_for(b, faces_b[0]["kps"])
                        if aligned_a is None or aligned_b is None:
                            error = True
                        else:
                            emb1 = wrapper.embed(aligned_a)
                            emb2 = wrapper.embed(aligned_b)
                            if emb1 is None or emb2 is None:
                                error = True
                else:
                    # aligned datasets (e.g. LFW-deepfunneled) – embed directly
                    emb1 = wrapper.embed(a)
                    emb2 = wrapper.embed(b)
                    if emb1 is None or emb2 is None:
                        error = True

        if error:
            # Force error into FN/FP for ROC consistency
            if label == 1:
                sim = -1.0  # FN
            else:
                sim = 2.0  # FP
        else:
            sim = cosine_similarity(emb1, emb2)

        sims.append(sim)
        labels.append(label)

    # ---------- everything below this stays as you had it ----------
    sims = np.array(sims, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    pos_count = int(np.sum(labels == 1))
    neg_count = int(np.sum(labels == 0))

    # --- guard: ROC requires both classes ---
    if pos_count == 0 or neg_count == 0:
        print(
            json.dumps(
                {
                    "error": "ROC cannot be computed: dataset contains only one class after evaluation.",
                    "pos_pairs": pos_count,
                    "neg_pairs": neg_count,
                }
            )
        )
        return

    fpr, tpr, thresholds = roc_curve(labels, sims)
    roc_auc = auc(fpr, tpr)

    # --- Best Threshold (Youden's J statistic) ---
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    print(f"[THRESHOLD] Best threshold (Youden J): {best_threshold:.4f}")

    # Compute EER
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0

    # Compute TAR @ FAR = 1e-3
    target_far = 1e-3
    idx = np.searchsorted(fpr, target_far, side="right") - 1
    if 0 <= idx < len(tpr):
        tar_at_far = tpr[idx]
    else:
        tar_at_far = float("nan")

    # ---- Build export pair records ----
    pair_records = []
    used_identities = {}

    for (img1, img2, label), sim in zip(pairs, sims):
        person1 = os.path.basename(os.path.dirname(img1))
        person2 = os.path.basename(os.path.dirname(img2))

        used_identities.setdefault(person1, set()).add(os.path.basename(img1))
        used_identities.setdefault(person2, set()).add(os.path.basename(img2))

        pair_records.append(
            {
                "img1": img1.replace(dataset_path + os.sep, ""),
                "img2": img2.replace(dataset_path + os.sep, ""),
                "label": "pos" if label == 1 else "neg",
                "similarity": float(sim),
            }
        )

    export = {
        "meta": {
            "model": model_name,
            "dataset": os.path.basename(dataset_path),
            "test_name": "ROC (Image)",
            "pairs_evaluated": int(len(labels)),
            "pos_pairs": pos_count,
            "neg_pairs": neg_count,
            "start_identity": start_identity,
            "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
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

    # ---- Save Export JSON ----
    export_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "exports",
        f"{model_name}_roc_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
    )

    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    with open(export_path, "w") as f:
        json.dump(export, f, indent=2)

    print(f"[EXPORTED] -> {os.path.abspath(export_path)}")

    # Save ROC plot
    save_path = os.path.join(os.path.dirname(__file__), "roc_result.png")
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {model_name}")
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    sys.stdout.write(
        json.dumps(
            {
                "kind": "roc_image",
                "path": save_path,
                "auc": float(roc_auc),
                "eer": float(eer),
                "best_threshold": float(best_threshold),
                "tar_far_1e3": float(tar_at_far),
                "pairs_tested": int(len(labels)),
                "pos_pairs": pos_count,
                "neg_pairs": neg_count,
            }
        )
        + "\n"
    )
    sys.stdout.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--iters", type=int, default=300)
    args = parser.parse_args()
    run_roc(args.model, args.dataset, args.start, args.iters)
