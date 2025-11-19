# trying to fix roc_lfw_pairs.py code, because results for facenet and adaface are really low


import os
import sys
import json
from datetime import datetime

import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc

# make project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from connector import load_model


# ================================================================
# Utility
# ================================================================
def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_lfw_pairs_with_folds(pairs_file, dataset_path):
    """
    Loads:
        - 6000 LFW image pairs
        - fold IDs 0..9
    According to official pairs.txt format.
    """
    pairs = []
    fold_ids = []

    with open(pairs_file, "r") as f:
        lines = f.read().strip().split("\n")

    num_folds, pairs_per_fold = map(int, lines[0].split())
    idx = 1

    for fold in range(num_folds):
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


# ================================================================
# 10-FOLD ACCURACY (OFFICIAL)
# ================================================================
def compute_10fold_protocol(similarities, labels, fold_ids):
    num_folds = int(fold_ids.max()) + 1
    fold_thresholds = []
    fold_accuracies = []

    for fold in range(num_folds):
        train_mask = fold_ids != fold
        test_mask = fold_ids == fold

        sims_train = similarities[train_mask]
        labels_train = labels[train_mask]
        sims_test = similarities[test_mask]
        labels_test = labels[test_mask]

        # pick best threshold using TRAINING FOLDS ONLY
        cand_thr = np.unique(sims_train)
        best_thr = None
        best_acc = -1

        for t in cand_thr:
            preds = (sims_train >= t).astype(np.int32)
            acc = np.mean(preds == labels_train)
            if acc > best_acc:
                best_acc = acc
                best_thr = t

        # evaluate on this fold
        preds_test = (sims_test >= best_thr).astype(np.int32)
        test_acc = np.mean(preds_test == labels_test)

        fold_thresholds.append(float(best_thr))
        fold_accuracies.append(float(test_acc))

        print(f"[Fold {fold}] thr={best_thr:.4f}, acc={test_acc*100:.2f}%")

    return (
        np.array(fold_thresholds, dtype=np.float32),
        np.array(fold_accuracies, dtype=np.float32),
        float(np.mean(fold_accuracies)),
        float(np.std(fold_accuracies)),
    )


# ================================================================
# MAIN LFW EVALUATION
# ================================================================
def run_lfw_10fold(model_name, dataset_root, pairs_file):
    print(f"--- LFW 10-Fold Evaluation ---")
    print(f"Model   : {model_name}")
    print(f"Dataset : {dataset_root}")

    # Always use deepfunneled images if present
    if os.path.isdir(os.path.join(dataset_root, "lfw-deepfunneled")):
        dataset_root = os.path.join(dataset_root, "lfw-deepfunneled")

    # Load wrapper
    wrapper = load_model(model_name)
    name = getattr(wrapper, "name", "").lower()
    is_arcface = name == "arcface"
    is_adaface = name == "adaface"
    is_facenet = name == "facenet"

    # Load pairs and fold IDs
    pairs, fold_ids = load_lfw_pairs_with_folds(pairs_file, dataset_root)

    sims = []
    labels = []

    # ============================================================
    # OFFICIAL MODEL-SPECIFIC ALIGNMENT BEHAVIOR
    # ============================================================
    print(
        "\n[Eval] Using model-specific alignment rules:\n"
        f"- ArcFace  → wrapper.get_embedding(path)\n"
        f"- AdaFace  → wrapper.embed(img) on deepfunneled\n"
        f"- FaceNet  → detect + align_for\n"
    )

    for idx, (img1, img2, label) in enumerate(pairs, 1):
        if idx % 300 == 0 or idx == 1 or idx == len(pairs):
            print(f"[Eval] {idx}/{len(pairs)} ({idx/len(pairs)*100:.1f}%)")

        a = cv2.imread(img1)
        b = cv2.imread(img2)
        if a is None or b is None:
            sim = -1 if label == 1 else 2
            sims.append(sim)
            labels.append(label)
            continue

        emb1 = emb2 = None
        error = False

        try:
            if is_arcface:
                emb1 = wrapper.get_embedding(img1)
                emb2 = wrapper.get_embedding(img2)

            elif is_adaface:
                # deepfunneled images are already aligned
                emb1 = wrapper.embed(a)
                emb2 = wrapper.embed(b)

            elif is_facenet:
                faces_a = wrapper.detector.detect(a)
                faces_b = wrapper.detector.detect(b)

                if not faces_a or not faces_b:
                    error = True
                else:
                    kps_a = faces_a[0]["kps"]
                    kps_b = faces_b[0]["kps"]
                    crop_a = wrapper.detector.align_for(a, kps_a)
                    crop_b = wrapper.detector.align_for(b, kps_b)

                    if crop_a is None or crop_b is None:
                        error = True
                    else:
                        emb1 = wrapper.embed(crop_a)
                        emb2 = wrapper.embed(crop_b)

            else:
                # generic fallback
                emb1 = wrapper.embed(a)
                emb2 = wrapper.embed(b)

        except Exception:
            error = True

        # scoring
        if error or emb1 is None or emb2 is None:
            sim = -1 if label == 1 else 2
        else:
            sim = cosine_similarity(emb1, emb2)

        sims.append(sim)
        labels.append(label)

    sims = np.array(sims, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # ============================================================
    # Compute metrics
    # ============================================================
    print("\n[Eval] Computing 10-fold metrics...")
    thr, accs, mean_acc, std_acc = compute_10fold_protocol(sims, labels, fold_ids)

    print("\n[Eval] Computing ROC / AUC / EER...")
    fpr, tpr, roc_thr = roc_curve(labels, sims)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = 0.5 * (fpr[eer_idx] + fnr[eer_idx])

    # ============================================================
    # Export
    # ============================================================
    export_dir = os.path.join(os.path.dirname(__file__), "exports")
    os.makedirs(export_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{name}_lfw10_{timestamp}"

    json_data = {
        "model": model_name,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "per_fold_accuracy": accs.tolist(),
        "per_fold_threshold": thr.tolist(),
        "auc": float(roc_auc),
        "eer": float(eer),
    }

    json_path = os.path.join(export_dir, base + ".json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"[Export] Saved JSON: {json_path}")

    # Save ROC PNG
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(fpr, tpr, label=f"{model_name} AUC={roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"LFW ROC - {model_name}")
        plt.legend()
        plt.grid(True)

        png_path = os.path.join(export_dir, base + ".png")
        plt.savefig(png_path, dpi=160, bbox_inches="tight")
        plt.close()

        print(f"[Export] Saved ROC PNG: {png_path}")

    except Exception as e:
        print(f"[WARN] Could not save ROC PNG ({e})")

    # final summary
    print("\n=== FINAL RESULTS ===")
    print(json.dumps(json_data, indent=2))

    return json_data


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pairs", required=True)
    args = parser.parse_args()

    run_lfw_10fold(args.model, args.dataset, args.pairs)
