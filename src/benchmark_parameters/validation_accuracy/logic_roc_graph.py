import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import sys
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ✅ Correct path: go up two folders to reach src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
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
        p1, p2 = people[i], people[i + 1]

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
        if imgs1 and imgs2:
            neg_pairs.append(
                (
                    os.path.join(dataset_path, p1, imgs1[0]),
                    os.path.join(dataset_path, p2, imgs2[0]),
                    0,
                )
            )

    pairs = pos_pairs[: max_pairs // 2] + neg_pairs[: max_pairs // 2]
    return pairs[:max_pairs]


def run_roc(model_name, dataset_path, iters=300):
    print(f"[ROC] Model: {model_name}")
    print(f"[ROC] Dataset: {dataset_path}")
    print(f"[ROC] Pairs: {iters}")

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

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, sims)
    roc_auc = auc(fpr, tpr)

    # --------- ✅ Compute EER ---------
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0

    # --------- ✅ Compute TAR @ FAR = 1e-3 ---------
    target_far = 1e-3
    idx = np.searchsorted(fpr, target_far, side="right") - 1
    if 0 <= idx < len(tpr):
        tar_at_far = tpr[idx]
    else:
        tar_at_far = float("nan")

    # Save ROC PNG
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

    # --------- ✅ Output JSON to GUI ---------
    print(
        json.dumps(
            {
                "kind": "roc_image",
                "path": save_path,
                "auc": float(roc_auc),
                "eer": float(eer),
                "tar_far_1e3": float(tar_at_far),
                "pairs_tested": int(len(labels)),
                "model": model_name,
                "dataset": os.path.basename(dataset_path),
            }
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--iters", type=int, default=300)
    args = parser.parse_args()

    run_roc(args.model, args.dataset, args.iters)
