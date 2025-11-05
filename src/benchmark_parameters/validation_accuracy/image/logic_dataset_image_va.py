# logic_dataset_image_va.py
import os
import cv2
import numpy as np
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from connector import load_model


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


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

    for person in people:
        imgs = sorted(
            [
                os.path.join(dataset_path, person, f)
                for f in os.listdir(os.path.join(dataset_path, person))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        if len(imgs) >= 2:
            pos_pairs.append((imgs[0], imgs[1], 1))

    for i in range(len(people) - 1):
        p1 = people[i]
        p2 = people[i + 1]

        imgs1 = sorted(
            [
                f
                for f in os.listdir(os.path.join(dataset_path, p1))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        imgs2 = sorted(
            [
                f
                for f in os.listdir(os.path.join(dataset_path, p2))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        if not imgs1 or not imgs2:
            continue

        neg_pairs.append(
            (
                os.path.join(dataset_path, p1, imgs1[0]),
                os.path.join(dataset_path, p2, imgs2[0]),
                0,
            )
        )

    pairs = pos_pairs[: max_pairs // 2] + neg_pairs[: max_pairs // 2]
    return pairs[:max_pairs]


def run_logic(model_name, iters=300, frame_h=None, frame_w=None, dataset_path=None):
    print(f"[ROC] Model: {model_name}")
    print(f"[ROC] Dataset: {dataset_path}")

    wrapper = load_model(model_name)

    if os.path.isdir(os.path.join(dataset_path, "lfw-deepfunneled")):
        dataset_path = os.path.join(dataset_path, "lfw-deepfunneled")

    pairs = collect_pairs(dataset_path, max_pairs=iters)

    sims = []
    labels = []

    for img1, img2, label in tqdm(pairs):
        a = cv2.imread(img1)
        b = cv2.imread(img2)
        if a is None or b is None:
            continue
        emb1 = wrapper.embed(a)
        emb2 = wrapper.embed(b)
        if emb1 is None or emb2 is None:
            continue
        sims.append(cosine_similarity(emb1, emb2))
        labels.append(label)

    sims = np.array(sims)
    labels = np.array(labels)

    fpr, tpr, _ = roc_curve(labels, sims)
    auc_val = auc(fpr, tpr)

    # --- EER (point where FAR == FRR) ---
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fnr[eer_idx] + fpr[eer_idx]) / 2.0)

    # --- TAR @ FAR = 1e-3 ---
    target_far = 1e-3
    # fpr is monotonic from roc_curve, so interpolation is OK
    tar_at_far = float(np.interp(target_far, fpr, tpr, left=tpr[0], right=tpr[-1]))

    export_path = os.path.join(os.path.dirname(__file__), "last_roc.png")

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC: {model_name} on LFW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(export_path)
    plt.close()

    print(f"[ROC] Saved to: {export_path}")

    result = {
        "kind": "roc_image",
        "path": export_path,
        "auc": float(auc_val),
        "eer": float(eer),  # <--- NEW
        "tar_far_1e3": float(tar_at_far),  # <--- NEW
        "pairs_tested": len(labels),
        "model": model_name,
        "dataset": os.path.basename(dataset_path),
    }

    print(json.dumps(result))
