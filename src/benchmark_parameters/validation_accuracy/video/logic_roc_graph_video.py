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

# ---------------------------------------------------------
# Shared helpers (import from confusion-matrix implementation)
# ---------------------------------------------------------
try:
    # Case 1: imported as part of the package
    from .logic_confusion_matrix_video import (
        cosine_similarity,
        _resolve_video_root,
        collect_pairs,
    )
except ImportError:
    # Case 2: executed as a top-level script (no known parent package)
    from benchmark_parameters.validation_accuracy.video.logic_confusion_matrix_video import (  # type: ignore  # noqa: E501
        cosine_similarity,
        _resolve_video_root,
        collect_pairs,
    )


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
    failed_pairs = 0

    for i, (img1, img2, label) in enumerate(pairs, start=1):
        # progress for GUI
        sys.stdout.write(
            json.dumps({"_type": "progress", "progress": i, "total": total}) + "\n"
        )
        sys.stdout.flush()

        error = False

        a = cv2.imread(img1)
        b = cv2.imread(img2)

        error = False

        if a is None or b is None:
            error = True
        else:
            # --- SAFE DETECTION ---
            try:
                faces_a = wrapper.detector.detect(a)
                faces_b = wrapper.detector.detect(b)
            except Exception:
                faces_a, faces_b = [], []

            if not faces_a or not faces_b:
                error = True
            else:
                # --- ALIGNMENT ---
                aligned_a = wrapper.detector.align_for(a, faces_a[0]["kps"])
                aligned_b = wrapper.detector.align_for(b, faces_b[0]["kps"])

                if aligned_a is None or aligned_b is None:
                    error = True
                else:
                    # --- EMBEDDING ---
                    emb1 = wrapper.embed(aligned_a)
                    emb2 = wrapper.embed(aligned_b)
                    if emb1 is None or emb2 is None:
                        error = True

        # --- FORCE SCORE ON FAIL ---
        if error:
            failed_pairs += 1
            sim = -1.0 if label == 1 else 2.0
        else:
            sim = cosine_similarity(emb1, emb2)

        # Keep similarity in a sane numeric range
        sim = float(np.clip(sim, -1.0, 2.0))

        if error:
            failed_pairs += 1
            # Worst-case scores: always misclassified across [0,1] thresholds
            if label == 1:
                sim = -1.0  # positive pair → always below threshold → FN
            else:
                sim = 2.0  # negative pair → always above threshold → FP
        else:
            sim = cosine_similarity(emb1, emb2)

        sims.append(sim)
        labels.append(label)

        # track identities used (even for failed pairs)
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
                "error": bool(error),
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
            "pairs_evaluated": int(len(labels)),
            "pairs_built": int(total),
            "failed_pairs": int(failed_pairs),
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
                "pairs_built": int(total),
                "failed_pairs": int(failed_pairs),
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
