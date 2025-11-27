import os

os.environ["MPLBACKEND"] = "Agg"

import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)

from connector import load_model
from models.wrap_facedetection import FaceDetectorAligner
from models.wrap_facedetection import align_face_5pts, REF_5PTS_112, REF_5PTS_160

# -------- InsightFace alignment (for ArcFace only) --------
try:
    from insightface.utils import face_align

    INSIGHT_AVAILABLE = True
except:
    INSIGHT_AVAILABLE = False


# ==========================================================
#                      COSINE SIMILARITY
# ==========================================================
def cosine_similarity(a, b):
    a = np.asarray(a, np.float32).flatten()
    b = np.asarray(b, np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else -1.0


# ==========================================================
#                     LOAD PAIRS
# ==========================================================
def load_pairs(pairs_file, dataset_root):
    pairs = []
    with open(pairs_file, "r") as f:
        lines = f.read().strip().split("\n")

    num_folds, pairs_per_fold = map(int, lines[0].split())
    idx = 1

    for _fold in range(num_folds):
        for _ in range(pairs_per_fold):
            p, i1, i2 = lines[idx].split()
            pairs.append(
                (
                    os.path.join(dataset_root, p, i1),
                    os.path.join(dataset_root, p, i2),
                    1,
                )
            )
            idx += 1

        for _ in range(pairs_per_fold):
            p1, i1, p2, i2 = lines[idx].split()
            pairs.append(
                (
                    os.path.join(dataset_root, p1, i1),
                    os.path.join(dataset_root, p2, i2),
                    0,
                )
            )
            idx += 1

    return pairs


# ==========================================================
#              MODEL-SPECIFIC EMBEDDING (MATCH ROC)
# ==========================================================
def extract_embedding(model_name, wrapper, detector, img):
    dets = detector.detect(img)
    if not dets:
        return None

    kps = dets[0]["kps"]

    # ------------------------------------------------------
    # ARC FACE → InsightFace 5pt norm_crop (112×112)
    # ------------------------------------------------------
    if model_name.lower() == "arcface":
        kps_ins = np.array(
            [
                kps[1],  # right eye
                kps[0],  # left eye
                kps[2],  # nose
                kps[3],  # left mouth
                kps[4],  # right mouth
            ],
            dtype=np.float32,
        )

        aligned = face_align.norm_crop(img, kps_ins, image_size=112)
        return wrapper.embed_aligned(aligned)

    # ------------------------------------------------------
    # ADAFACE / FACENET → Your 5pt 160×160 aligner
    # ------------------------------------------------------
    aligned = align_face_5pts(img, kps, out_size=(160, 160))
    if aligned is None:
        return None

    emb = wrapper.embed(aligned)
    if emb is None:
        return None

    # L2 normalize (necessary for AdaFace + FaceNet)
    n = np.linalg.norm(emb)
    if n > 0:
        emb = emb / n

    return emb


# ==========================================================
#                     CONFUSION MATRIX PLOT
# ==========================================================
def plot_confusion_matrix_lfw(cm, model_name, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=18)
    ax.set_xlabel("Predicted label", fontsize=16)
    ax.set_ylabel("Actual label", fontsize=16)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"], fontsize=14)
    ax.set_yticklabels(["Negative", "Positive"], fontsize=14)

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=15,
            )

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ==========================================================
#                         MAIN
# ==========================================================
def run_confusion(model_name, dataset_root, pairs_file):
    print(f"[INFO] Confusion evaluation for: {model_name}")

    wrapper = load_model(model_name)
    detector = FaceDetectorAligner(device="cpu")

    pairs = load_pairs(pairs_file, dataset_root)

    sims = []
    labels = []

    # -------------------- Compute similarities --------------------
    for img1, img2, lab in pairs:
        a = cv2.imread(img1)
        b = cv2.imread(img2)

        if a is None or b is None:
            continue

        emb1 = extract_embedding(model_name, wrapper, detector, a)
        emb2 = extract_embedding(model_name, wrapper, detector, b)

        if emb1 is None or emb2 is None:
            continue

        sims.append(cosine_similarity(emb1, emb2))
        labels.append(lab)

    sims = np.array(sims, np.float32)
    labels = np.array(labels, np.int32)

    # -------------------- Find best threshold --------------------
    thrs = np.unique(sims)
    best_thr, best_acc = 0, -1
    for t in thrs:
        preds = (sims >= t).astype(np.int32)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc = acc
            best_thr = t

    preds = (sims >= best_thr).astype(np.int32)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    print(f"[THRESHOLD] best={best_thr:.4f}")
    print(f"[ACCURACY] {best_acc*100:.2f}%")
    print(f"TN={tn}, FP={fp}")
    print(f"FN={fn}, TP={tp}")

    # -------------------- Save JSON --------------------
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_clean = model_name.lower().replace(" ", "_")

    summary = {
        "model": model_name,
        "pairs": len(labels),
        "best_threshold": float(best_thr),
        "accuracy": float(best_acc),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    json_path = f"{model_clean}_custom_confusion_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Saved JSON: {json_path}")

    # -------------------- Save PNG --------------------
    cm = np.array([[tn, fp], [fn, tp]])
    png_path = f"{model_clean}_custom_confusion_{timestamp}.png"
    plot_confusion_matrix_lfw(cm, model_name, png_path)
    print(f"[OK] Saved PNG: {png_path}")


# ----------------------------- ENTRY POINT -----------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--pairs", required=True)
    args = p.parse_args()

    run_confusion(args.model, args.dataset, args.pairs)
