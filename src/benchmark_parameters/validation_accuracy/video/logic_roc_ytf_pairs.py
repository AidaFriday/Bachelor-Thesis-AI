# logic_roc_ytf_pairs.py
#
# Computes ROC metrics for YTF (YouTube Faces) across all 10 official folds
# using precomputed similarity scores and labels. Exports ROC metrics JSON
# and saves a ROC curve PNG.

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# allow importing from project root if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


def load_all_scores_labels(exports_dir: Path, model: str, stamp: str):
    """
    Load and concatenate scores/labels for all 10 YTF folds for a given model+stamp.

    Expected filenames:
      <model>_ytf_fold0_<stamp>_scores.npy
      <model>_ytf_fold0_<stamp>_labels.npy
      ...
      <model>_ytf_fold9_<stamp>_scores.npy
      <model>_ytf_fold9_<stamp>_labels.npy
    """
    scores_list = []
    labels_list = []

    for fold in range(10):
        prefix = f"{model}_ytf_fold{fold}_{stamp}"
        scores_path = exports_dir / f"{prefix}_scores.npy"
        labels_path = exports_dir / f"{prefix}_labels.npy"

        if not scores_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Missing files for fold {fold}: {scores_path}, {labels_path}"
            )

        scores_list.append(np.load(scores_path))
        labels_list.append(np.load(labels_path))

    scores = np.concatenate(scores_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return scores, labels


def run_roc_ytf(model_name: str, stamp: str, exports_dir: str | None = None):
    """
    Compute ROC metrics on all 10 folds (5000 pairs) of YTF
    using precomputed scores/labels.
    """
    if exports_dir is None:
        exports_dir = Path(__file__).resolve().parents[2] / "exports"
    else:
        exports_dir = Path(exports_dir)

    scores, labels = load_all_scores_labels(exports_dir, model_name, stamp)

    # --- basic sanity check ---
    pos_count = int((labels == 1).sum())
    neg_count = int((labels == 0).sum())
    print(f"[YTF ROC] pos_pairs={pos_count}, neg_pairs={neg_count}")
    print(
        f"[YTF ROC] scores range: [{float(scores.min()):.4f}, {float(scores.max()):.4f}]"
    )

    if pos_count == 0 or neg_count == 0:
        raise ValueError(
            f"ROC cannot be computed: only one class present "
            f"(pos={pos_count}, neg={neg_count})."
        )

    # --- ROC ---
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Best threshold: Youden's J statistic
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thr = thresholds[best_idx]

    # EER
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0

    # TAR @ FAR = 1e-3
    target_far = 1e-3
    idx = np.searchsorted(fpr, target_far, side="right") - 1
    if 0 <= idx < len(tpr):
        tar_far_1e3 = tpr[idx]
    else:
        tar_far_1e3 = float("nan")

    metrics = {
        "kind": "roc_ytf",
        "model": model_name,
        "dataset": "YTF",
        "auc": float(roc_auc),
        "eer": float(eer),
        "best_threshold": float(best_thr),
        "tar_far_1e3": float(tar_far_1e3),
        "pairs": int(len(labels)),
        "pos_pairs": pos_count,
        "neg_pairs": neg_count,
    }

    # --- export JSON ---
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = exports_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{model_name}_ytf_roc_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # --- export ROC PNG (overwrites same file each time) ---
    png_path = Path(__file__).with_name("roc_ytf_result.png")
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"YTF ROC – {model_name}")
    plt.legend(loc="lower right")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[YTF ROC] JSON -> {json_path}")
    print(f"[YTF ROC] PNG  -> {png_path}")
    print(json.dumps(metrics))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="facenet / arcface / adaface ..."
    )
    parser.add_argument(
        "--stamp",
        required=True,
        help="timestamp from export_ytf_pairs (e.g. 20251110-221503)",
    )
    parser.add_argument(
        "--exports-dir",
        default=None,
        help="override exports directory (optional)",
    )
    args = parser.parse_args()

    run_roc_ytf(args.model, args.stamp, args.exports_dir)
