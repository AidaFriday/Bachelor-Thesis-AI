"""
UNIFIED ROC EVALUATOR
---------------------
Supports ArcFace, AdaFace, FaceNet in ONE pipeline.
Uses YOLOv5-face for detection for ALL models.
Uses correct alignment for each model:
- ArcFace → InsightFace 5-pt norm_crop
- AdaFace/FaceNet → your REF_5PTS_160 alignment
"""

import os

os.environ["MPLBACKEND"] = "Agg"

import sys
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from datetime import datetime

# project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)

from connector import load_model
from models.wrap_facedetection import FaceDetectorAligner  # your YOLO detector

# ---------- Your alignment utilities ----------
from models.wrap_facedetection import align_face_5pts, REF_5PTS_112, REF_5PTS_160

try:
    import insightface
    from insightface.utils import face_align

    INSIGHT_AVAILABLE = True
except:
    INSIGHT_AVAILABLE = False


# ======================================================
#              HELPER: COSINE SIMILARITY
# ======================================================
def cosine_similarity(a, b):
    a = np.asarray(a, np.float32).flatten()
    b = np.asarray(b, np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else -1.0


# ======================================================
#          LOAD PAIRS (Your original LFW-style)
# ======================================================
def load_pairs_with_folds(pairs_file, dataset_root):
    pairs = []
    fold_ids = []

    with open(pairs_file) as f:
        lines = f.read().strip().split("\n")

    num_folds, pairs_per_fold = map(int, lines[0].split())
    idx = 1

    for fold in range(num_folds):

        # positive
        for _ in range(pairs_per_fold):
            person, img1, img2 = lines[idx].split()
            img1p = os.path.join(dataset_root, person, img1)
            img2p = os.path.join(dataset_root, person, img2)
            pairs.append((img1p, img2p, 1))
            fold_ids.append(fold)
            idx += 1

        # negative
        for _ in range(pairs_per_fold):
            p1, img1, p2, img2 = lines[idx].split()
            img1p = os.path.join(dataset_root, p1, img1)
            img2p = os.path.join(dataset_root, p2, img2)
            pairs.append((img1p, img2p, 0))
            fold_ids.append(fold)
            idx += 1

    return pairs, np.array(fold_ids, np.int32)


# ======================================================
#              FOLD ACCURACY
# ======================================================
def compute_fold_accuracy(sims, labels, fold_ids):
    if len(sims) == 0:
        return np.array([]), np.array([]), 0.0, 0.0

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

        if len(sims_train) == 0 or len(sims_test) == 0:
            thresholds.append(0.0)
            accuracies.append(0.0)
            continue

        unique_thr = np.unique(sims_train)

        best_thr = None
        best_acc = -1

        for t in unique_thr:
            preds = (sims_train >= t).astype(np.int32)
            acc = np.mean(preds == labels_train)
            if acc > best_acc:
                best_acc = acc
                best_thr = t

        preds_test = (sims_test >= best_thr).astype(np.int32)
        test_acc = np.mean(preds_test == labels_test)

        thresholds.append(float(best_thr))
        accuracies.append(float(test_acc))

    thresholds = np.array(thresholds, np.float32)
    accuracies = np.array(accuracies, np.float32)

    return thresholds, accuracies, float(np.mean(accuracies)), float(np.std(accuracies))


# ======================================================
#           MODEL-SPECIFIC EMBEDDING HANDLER
# ======================================================
def extract_embedding(model_name, wrapper, detector, img):

    # detect using YOLOv5-face always
    dets = detector.detect(img)
    if not dets:
        return None

    kps = dets[0]["kps"]

    # --------------------------------------
    # ARC FACE (requires InsightFace alignment)
    # --------------------------------------
    if model_name.lower() == "arcface":
        # YOLO gives: left_eye, right_eye, nose, left_mouth, right_mouth
        # InsightFace wants: right_eye, left_eye, nose, left_mouth, right_mouth
        kps_insight = np.array(
            [
                kps[1],  # right_eye
                kps[0],  # left_eye
                kps[2],  # nose
                kps[3],  # left_mouth
                kps[4],  # right_mouth
            ],
            dtype=np.float32,
        )

        # insightface alignment
        aligned = face_align.norm_crop(img, kps_insight, image_size=112)

        emb = wrapper.embed_aligned(aligned)
        return emb

    # --------------------------------------
    # ADAFACE / FACENET / OTHER MODELS
    # --------------------------------------
    aligned = align_face_5pts(img, kps, out_size=(160, 160))
    if aligned is None:
        return None

    emb = wrapper.embed(aligned)

    # L2 normalize (important!)
    if emb is not None:
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

    return emb


# ======================================================
#                     MAIN EVALUATION
# ======================================================
def run_custom_roc(model_name, dataset_root, pairs_file):
    print(f"[INFO] Evaluating model: {model_name}")

    wrapper = load_model(model_name)
    detector = FaceDetectorAligner(device="cpu")  # YOLOv5-face

    # load pairs
    pairs, fold_ids = load_pairs_with_folds(pairs_file, dataset_root)

    sims = []
    labels = []
    valid_fold_ids = []
    num_failed = 0

    # --------------------------------------------------
    #       PROCESS ALL PAIRS
    # --------------------------------------------------
    for (img1, img2, lab), fold in zip(pairs, fold_ids):

        a = cv2.imread(img1)
        b = cv2.imread(img2)
        if a is None or b is None:
            num_failed += 1
            continue

        emb1 = extract_embedding(model_name, wrapper, detector, a)
        emb2 = extract_embedding(model_name, wrapper, detector, b)

        if emb1 is None or emb2 is None:
            num_failed += 1
            continue

        sim = cosine_similarity(emb1, emb2)

        # ✅ LFW-correct exclusion logic
        sims.append(sim)
        labels.append(lab)
        valid_fold_ids.append(fold)

    sims = np.array(sims, np.float32)
    labels = np.array(labels, np.int32)
    fold_ids = np.array(valid_fold_ids, np.int32)

    # ==================================================
    # Pair statistics
    # ==================================================
    pairs_total = len(pairs)
    pairs_used = len(labels)
    pairs_failed = num_failed
    failure_rate = pairs_failed / (pairs_used + pairs_failed)

    print(
        f"[INFO] pairs_total={pairs_total}, "
        f"pairs_used={pairs_used}, "
        f"pairs_failed={pairs_failed}, "
        f"failure_rate={failure_rate:.4f}"
    )

    # --------------------------------------------------
    #       FOLD ACC
    # --------------------------------------------------
    thresholds, accs, mean_acc, std_acc = compute_fold_accuracy(sims, labels, fold_ids)

    print("\n--------------- RESULTS ----------------")
    print(f"3-fold mean accuracy: {mean_acc*100:.4f}% ± {std_acc*100:.4f}%")

    # --------------------------------------------------
    #       GLOBAL ROC
    # --------------------------------------------------
    if len(sims) > 0:
        fpr, tpr, thr = roc_curve(labels, sims)
        auc_val = auc(fpr, tpr)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = 0.5 * (fpr[idx] + fnr[idx])

        best_idx = np.argmax(tpr - fpr)
        best_threshold = float(thr[best_idx])

        print(f"AUC: {auc_val:.4f}")
        print(f"EER: {eer*100:.2f}%")
        print(f"Best threshold: {best_threshold}")
    else:
        auc_val = eer = best_threshold = 0.0
        print("WARNING: no valid pairs found!")

    # --------------------------------------------------
    #       PLOT
    # --------------------------------------------------
    if len(sims) > 0:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.4f})", linewidth=2.2)

        plt.plot([0, 1], [0, 1], linestyle="--", color="#555", linewidth=1.5)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.grid(True)
        plt.legend(loc="lower right", fontsize=12)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # --------------------------------------------------
        #       SAVE PNG + JSON INTO EXPORTS/ROC FOLDER
        # --------------------------------------------------

        export_dir = os.path.join(os.path.dirname(__file__), "exports", "roc")
        os.makedirs(export_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{model_name.lower()}_customroc_{timestamp}"

        out_png = os.path.join(export_dir, base + ".png")
        out_json = os.path.join(export_dir, base + ".json")

        # Save ROC PNG
        if len(sims) > 0:
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[OK] Saved ROC PNG: {out_png}")

        # Save JSON
        summary = {
            "kind": "lfw_10fold_roc",
            "model": model_name,
            "dataset": os.path.basename(dataset_root),
            # --- pair statistics (LFW-style) ---
            "pairs_total": pairs_total,
            "pairs_valid": pairs_used,
            "pairs_failed": pairs_failed,
            "failure_rate": failure_rate,
            # --- ROC data ---
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(auc_val),
            "eer": float(eer),
            "best_threshold": float(best_threshold),
            # --- fold metrics ---
            "folds": int(fold_ids.max() + 1) if len(fold_ids) > 0 else 0,
            "per_fold_accuracy": accs.tolist(),
            "per_fold_threshold": thresholds.tolist(),
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
        }

        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[OK] Saved JSON: {out_json}")


# ======================================================
#                     ENTRY POINT
# ======================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pairs", required=True)
    args = parser.parse_args()

    run_custom_roc(args.model, args.dataset, args.pairs)
