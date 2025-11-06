import os
import cv2
import json
import numpy as np
from sklearn.metrics import roc_curve, auc

# ✅ Make project root importable
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from connector import load_model


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_lfw_pairs(pairs_file, dataset_path):
    pairs = []
    with open(pairs_file, "r") as f:
        lines = f.read().strip().split("\n")

    # first line example: "10 300"
    _, pairs_per_fold = map(int, lines[0].split())
    idx = 1

    for _ in range(10):  # 10 folds
        # positive pairs
        for _ in range(pairs_per_fold):
            name, n1, n2 = lines[idx].split()
            img1 = os.path.join(dataset_path, name, f"{name}_{int(n1):04d}.jpg")
            img2 = os.path.join(dataset_path, name, f"{name}_{int(n2):04d}.jpg")
            pairs.append((img1, img2, 1))
            idx += 1

        # negative pairs
        for _ in range(pairs_per_fold):
            name1, n1, name2, n2 = lines[idx].split()
            img1 = os.path.join(dataset_path, name1, f"{name1}_{int(n1):04d}.jpg")
            img2 = os.path.join(dataset_path, name2, f"{name2}_{int(n2):04d}.jpg")
            pairs.append((img1, img2, 0))
            idx += 1

    return pairs


def run_lfw_protocol(model_name, dataset_path, pairs_file):
    wrapper = load_model(model_name)
    pairs = load_lfw_pairs(pairs_file, dataset_path)

    sims = []
    labels = []

    total = len(pairs)
    for i, (img1, img2, label) in enumerate(pairs, start=1):

        # ---- LIVE TERMINAL LOG ----
        if i % 50 == 0 or i == 1:
            print(f"[LFW] Pair {i}/{total} ({(i/total)*100:.1f}%)")

        # (optional) GUI progress events, only if you already use GUI messaging elsewhere
        # sys.stdout.write(json.dumps({"_type": "progress", "progress": i, "total": total}) + "\n")
        # sys.stdout.flush()

        a = cv2.imread(img1)
        b = cv2.imread(img2)
        error = False

        if a is None or b is None:
            error = True
        else:
            fa = wrapper.detector.detect(a)
            fb = wrapper.detector.detect(b)
            if not fa or not fb:
                error = True
            else:
                a = wrapper.detector.align_for(a, fa[0]["kps"])
                b = wrapper.detector.align_for(b, fb[0]["kps"])
                if a is None or b is None:
                    error = True
                else:
                    e1 = wrapper.embed(a)
                    e2 = wrapper.embed(b)
                    if e1 is None or e2 is None:
                        error = True

        if error:
            sim = -1.0 if label == 1 else 2.0
        else:
            sim = cosine_similarity(e1, e2)

        sims.append(sim)
        labels.append(label)

    sims = np.array(sims)
    labels = np.array(labels)

    fpr, tpr, _ = roc_curve(labels, sims)
    roc_auc = auc(fpr, tpr)

    # ---- Compute EER ----
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0

    print(f"LFW Official Protocol AUC: {roc_auc:.4f}")
    print(f"EER: {eer:.4f}")

    return roc_auc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pairs", type=str, required=True)
    args = parser.parse_args()

    run_lfw_protocol(args.model, args.dataset, args.pairs)
