# benchmark_parameters/validation_accuracy/image/roc_lfw_pairs.py

import os
import cv2
import json
import numpy as np
from sklearn.metrics import roc_curve, auc
import sys

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
    Official-style 10-fold evaluation.
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

        cand_thr = np.unique(sims_train)

        best_thr = None
        best_acc = -1.0

        for t in cand_thr:
            preds = (sims_train >= t).astype(np.int32)
            acc = np.mean(preds == labels_train)
            if acc > best_acc:
                best_acc = acc
                best_thr = t

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
def run_lfw_protocol(model_name, dataset_path, pairs_file):
    # if user passed parent LFW folder, go into lfw-deepfunneled
    if os.path.isdir(os.path.join(dataset_path, "lfw-deepfunneled")):
        dataset_path = os.path.join(dataset_path, "lfw-deepfunneled")

    print(f"[LFW] Model   : {model_name}")
    print(f"[LFW] Dataset : {dataset_path}")
    print(f"[LFW] Pairs   : {pairs_file}")

    wrapper = load_model(model_name)
    model_name_lower = getattr(wrapper, "name", "").lower()
    is_arcface = model_name_lower == "arcface"
    is_adaface = model_name_lower == "adaface"

    pairs, fold_ids = load_lfw_pairs_with_folds(pairs_file, dataset_path)

    sims = []
    labels = []

    total = len(pairs)
    use_detection = dataset_needs_alignment(dataset_path)

    # ✅ For AdaFace we *want* detection+alignment, even on LFW-deepfunneled.
    if is_adaface:
        use_detection = True

    # ✅ Only non-ArcFace, non-AdaFace models may use embed_aligned on aligned datasets
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
                    # ✅ ArcFace: tested path using internal detection/alignment
                    emb1 = wrapper.get_embedding(img1)
                    emb2 = wrapper.get_embedding(img2)

                elif use_aligned:
                    # ✅ Other models that have embed_aligned on aligned datasets
                    emb1 = wrapper.embed_aligned(a)
                    emb2 = wrapper.embed_aligned(b)

                elif use_detection:
                    # ✅ Facenet / AdaFace / others on non-aligned datasets
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
                    # ✅ Fallback: already-aligned dataset, generic embed
                    emb1 = wrapper.embed(a)
                    emb2 = wrapper.embed(b)

            except Exception:
                error = True

        if emb1 is None or emb2 is None:
            error = True

        if error:
            # worst-case scores so they are misclassified for any thr in [0,1]
            sim = -1.0 if label == 1 else 2.0
        else:
            sim = cosine_similarity(emb1, emb2)

        sims.append(sim)
        labels.append(label)

    sims = np.array(sims, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    fold_ids = np.array(fold_ids, dtype=np.int32)

    # ---------- 10-fold accuracy ----------
    thresholds, accs, mean_acc, std_acc = compute_lfw_10fold_accuracy(
        sims, labels, fold_ids
    )

    # ---------- global ROC / AUC / EER ----------
    fpr, tpr, _ = roc_curve(labels, sims)
    roc_auc = auc(fpr, tpr)

    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = 0.5 * (fpr[eer_idx] + fnr[eer_idx])

    print(f"\nGlobal ROC AUC (all pairs): {roc_auc:.4f}")
    print(f"Global EER               : {eer*100:.2f}%")

    # ---------- JSON summary ----------
    result = {
        "kind": "lfw_10fold",
        "model": model_name,
        "dataset": os.path.basename(dataset_path),
        "pairs": int(len(labels)),
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "per_fold_accuracy": [float(a) for a in accs],
        "per_fold_threshold": [float(t) for t in thresholds],
        "auc": float(roc_auc),
        "eer": float(eer),
    }

    print("\n[LFW] JSON summary:")
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pairs", type=str, required=True)
    args = parser.parse_args()

    run_lfw_protocol(args.model, args.dataset, args.pairs)
