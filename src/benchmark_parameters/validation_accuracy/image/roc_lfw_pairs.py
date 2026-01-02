
#roc_lfw_pairs.py
import os
os.environ["MPLBACKEND"] = "Agg"
import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt

import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc

# make project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from connector import load_model  # noqa: E402


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def dataset_needs_alignment(dataset_path: str) -> bool:
    """
    Return False for already-aligned datasets (e.g. LFW-deepfunneled),
    True for "raw" datasets.
    """
    path = dataset_path.lower()
    if "lfw" in path and "deepfunneled" in path:
        return False
    return True


# ---------------------------------------------------------------------
# loader that also returns fold indices for each pair
# ---------------------------------------------------------------------
def load_lfw_pairs_with_folds(pairs_file, dataset_path):
    pairs = []
    fold_ids = []
    with open(pairs_file, "r") as f:
        lines = f.read().strip().split("\n")

    num_folds, pairs_per_fold = map(int, lines[0].split())
    idx = 1

    for fold in range(num_folds):  # 0..9
        # positive pairs
        for _ in range(pairs_per_fold):
            name, n1, n2 = lines[idx].split()
            img1 = os.path.join(dataset_path, name, f"{name}_{int(n1):04d}.jpg")
            img2 = os.path.join(dataset_path, name, f"{name}_{int(n2):04d}.jpg")
            pairs.append((img1, img2, 1))
            fold_ids.append(fold)
            idx += 1

        # negative pairs
        for _ in range(pairs_per_fold):
            name1, n1, name2, n2 = lines[idx].split()
            img1 = os.path.join(dataset_path, name1, f"{name1}_{int(n1):04d}.jpg")
            img2 = os.path.join(dataset_path, name2, f"{name2}_{int(n2):04d}.jpg")
            pairs.append((img1, img2, 0))
            fold_ids.append(fold)
            idx += 1

    return pairs, np.array(fold_ids, dtype=np.int32)


# ---------------------------------------------------------------------
# 10-fold LFW accuracy (threshold per fold)
# ---------------------------------------------------------------------
def compute_lfw_10fold_accuracy(sims, labels, fold_ids):
    """
    Official-style 10-fold evaluation:
      For each fold k:
        - train on folds != k → find best threshold (max accuracy)
        - evaluate on fold k with that threshold
    Returns:
      thresholds_per_fold, accuracies_per_fold, mean_acc, std_acc
    """
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

        # candidate thresholds = unique scores on training folds
        cand_thr = np.unique(sims_train)

        best_thr = None
        best_acc = -1.0

        for t in cand_thr:
            preds = (sims_train >= t).astype(np.int32)
            acc = np.mean(preds == labels_train)
            if acc > best_acc:
                best_acc = acc
                best_thr = t

        # evaluate on held-out fold
        test_preds = (sims_test >= best_thr).astype(np.int32)
        test_acc = np.mean(test_preds == labels_test)

        thresholds.append(float(best_thr))
        accuracies.append(float(test_acc))

        print(f"[Fold {k+1:02d}] thr={best_thr:.4f}, acc={test_acc*100:.2f}%")

    thresholds = np.array(thresholds, dtype=np.float32)
    accuracies = np.array(accuracies, dtype=np.float32)
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))

    print(
        f"\nLFW 10-fold accuracy: {mean_acc*100:.4f}% ± {std_acc*100:.4f}% "
        f"(mean ± std over {num_folds} folds)"
    )

    return thresholds, accuracies, mean_acc, std_acc


# ---------------------------------------------------------------------
# MAIN PROTOCOL
# ---------------------------------------------------------------------
def run_lfw_protocol(model_name, dataset_path, pairs_file, max_pairs=None):
    # if user passed parent LFW folder, go into lfw-deepfunneled
    if os.path.isdir(os.path.join(dataset_path, "lfw-deepfunneled")):
        dataset_path = os.path.join(dataset_path, "lfw-deepfunneled")

    print(f"[LFW] Model   : {model_name}")
    print(f"[LFW] Dataset : {dataset_path}")
    print(f"[LFW] Pairs   : {pairs_file}")

    # Load model (ArcFace, AdaFace, FaceNetOriginal, etc.)
    wrapper = load_model(model_name)

    model_name_lower = getattr(wrapper, "name", "").lower()
    is_arcface = model_name_lower == "arcface"
    is_adaface = model_name_lower == "adaface"

    pairs, fold_ids = load_lfw_pairs_with_folds(pairs_file, dataset_path)

    # optional speed-up for debugging
    if max_pairs is not None and max_pairs < len(pairs):
        pairs = pairs[:max_pairs]
        fold_ids = fold_ids[:max_pairs]
        print(f"[LFW] DEBUG: restricting to first {max_pairs} pairs")

    sims = []
    labels = []
    num_failed = 0 
    total = len(pairs)
    use_detection = dataset_needs_alignment(dataset_path)

    # only non-ArcFace, non-AdaFace models may use embed_aligned
    use_aligned = (
        (not use_detection)
        and (not is_arcface)
        and (not is_adaface)
        and hasattr(wrapper, "embed_aligned")
    )

    for i, (img1, img2, label) in enumerate(pairs, start=1):
        if i == 1 or i % 200 == 0 or i == total:
            print(f"[LFW] Pair {i}/{total} ({(i/total)*100:.1f}%)")

        a = cv2.imread(img1)
        b = cv2.imread(img2)

        error = False
        emb1 = emb2 = None

        if a is None or b is None:
            error = True
        else:
            try:
                if is_arcface:
                    # ArcFace: tested path using internal detection/alignment
                    emb1 = wrapper.get_embedding(img1)
                    emb2 = wrapper.get_embedding(img2)

                elif is_adaface:
                    # AdaFace: its own embed() expects an aligned crop.
                    # For LFW-deepfunneled we can treat the full image as a crop.
                    # Always treat LFW as aligned
                    emb1 = wrapper.embed(a)
                    emb2 = wrapper.embed(b)

                elif use_aligned:
                    # Other models that have embed_aligned on aligned datasets
                    emb1 = wrapper.embed_aligned(a)
                    emb2 = wrapper.embed_aligned(b)

                elif use_detection:
                    # Facenet / others on non-aligned datasets
                    faces_a = wrapper.detector.detect(a)
                    faces_b = wrapper.detector.detect(b)
                    if not faces_a or not faces_b:
                        error = True
                    else:
                        aligned_a = wrapper.detector.align_for(a, faces_a[0]["kps"])
                        aligned_b = wrapper.detector.align_for(b, faces_b[0]["kps"])
                        if aligned_a is None or aligned_b is None:
                            error = True
                        else:
                            emb1 = wrapper.embed(aligned_a)
                            emb2 = wrapper.embed(aligned_b)
                else:
                    # Fallback: already-aligned dataset, generic embed
                    emb1 = wrapper.embed(a)
                    emb2 = wrapper.embed(b)

            except Exception:
                error = True

        if emb1 is None or emb2 is None:
            error = True

        if error:
            num_failed += 1  
            sim = None
        else:
            sim = cosine_similarity(emb1, emb2)


        sims.append(sim)
        labels.append(label)

    sims = np.array(sims, dtype=object)
    labels = np.array(labels, dtype=np.int32)

    valid_mask = sims != None
    sims_valid = sims[valid_mask].astype(np.float32)
    labels_valid = labels[valid_mask]
    num_total = len(pairs)
    num_valid = len(sims_valid)




    fold_ids = np.array(fold_ids, dtype=np.int32)

    # ---------- 10-fold accuracy ----------
    thresholds, accs, mean_acc, std_acc = compute_lfw_10fold_accuracy(
        sims_valid,
        labels_valid,
        fold_ids[valid_mask]
)


    # ---------- global ROC / AUC / EER ----------
    fpr, tpr, roc_thresholds = roc_curve(labels_valid, sims_valid)
    roc_auc = auc(fpr, tpr)

    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = 0.5 * (fpr[eer_idx] + fnr[eer_idx])

    print(f"\nGlobal ROC AUC (all pairs): {roc_auc:.4f}")
    print(f"Global EER               : {eer*100:.2f}%")

    # --- Best global threshold (Youden's J statistic) ---
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(roc_thresholds[best_idx])
    print(f"[ROC] Best global threshold (Youden J): {best_threshold:.6f}")

    # ---------- JSON + PNG EXPORTS ----------
    exports_dir = os.path.join(os.path.dirname(__file__), "exports")
    os.makedirs(exports_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{model_name_lower}_roc_{timestamp}"

    # JSON summary (metrics + ROC points)
    roc_json = {
        "kind": "lfw_10fold_roc",
        "model": model_name,
        "dataset": os.path.basename(dataset_path),

        # --- pair statistics ---
        "pairs_total": int(num_total),
        "pairs_valid": int(num_valid),
        "pairs_failed": int(num_failed),

        # --- metrics (computed on valid pairs only) ---
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "per_fold_accuracy": [float(a) for a in accs],
        "per_fold_threshold": [float(t) for t in thresholds],
        "auc": float(roc_auc),
        "eer": float(eer),

        # --- ROC data ---
        "roc_fpr": [float(x) for x in fpr],
        "roc_tpr": [float(x) for x in tpr],
        "roc_thresholds": [float(x) for x in roc_thresholds],
        "best_threshold": float(best_threshold),
    }


    json_path = os.path.join(exports_dir, base_name + ".json")
    with open(json_path, "w") as f:
        json.dump(roc_json, f, indent=2)
    print(f"[ROC] Saved ROC JSON to: {json_path}")

    # PNG ROC curve
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--")  # diagonal
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name} on LFW")
        plt.legend(loc="lower right")
        plt.grid(True)

        png_path = os.path.join(exports_dir, base_name + ".png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[ROC] Saved ROC PNG to: {png_path}")

        # Send signal to GUI to display the ROC image ===
        print(
            json.dumps(
                {
                    "kind": "roc_image",
                    "path": png_path,
                    "model": model_name,
                    "dataset": os.path.basename(dataset_path),
                    "auc": float(roc_auc),
                    "eer": float(eer),
                    "pairs_tested": int(len(labels)),
                }
            )
        )

    except Exception as e:
        print(f"[ROC] WARNING: could not save ROC PNG ({e})")

    # ---------- print SHORT JSON summary to console ----------
    print("\n[LFW] JSON summary (no ROC arrays):")

    pretty_json = {
        k: v
        for k, v in roc_json.items()
        if not k.startswith("roc_")  # hide roc_fpr / roc_tpr / roc_thresholds
    }

    print(json.dumps(pretty_json, indent=2))

    return roc_json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pairs", type=str, required=True)
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="(debug) limit number of pairs evaluated",
    )
    args = parser.parse_args()

    run_lfw_protocol(args.model, args.dataset, args.pairs, max_pairs=args.max_pairs)
