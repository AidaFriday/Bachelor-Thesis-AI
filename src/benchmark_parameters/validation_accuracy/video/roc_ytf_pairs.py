import os
import sys
from pathlib import Path
import json
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# make project root importable (for consistency with rest of project)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


def load_all_scores_labels(exports_dir: Path, model: str, stamp: str):
    """
    Load and concatenate scores/labels for all 10 YTF folds.

    Expects files like:
      <model>_ytf_fold0_<stamp>_scores.npy
      <model>_ytf_fold0_<stamp>_labels.npy
      ...
      <model>_ytf_fold9_<stamp>_scores.npy
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
    if exports_dir is None:
        exports_dir = Path(__file__).resolve().parents[2] / "exports"
    else:
        exports_dir = Path(exports_dir)

    scores, labels = load_all_scores_labels(exports_dir, model_name, stamp)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Youden’s J
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thr = thresholds[best_idx]

    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0

    target_far = 1e-3
    idx = np.searchsorted(fpr, target_far, side="right") - 1
    tar_far_1e3 = tpr[idx] if 0 <= idx < len(tpr) else float("nan")

    metrics = {
        "kind": "roc_ytf",
        "model": model_name,
        "dataset": "YTF",
        "auc": float(roc_auc),
        "eer": float(eer),
        "best_threshold": float(best_thr),
        "tar_far_1e3": float(tar_far_1e3),
        "pairs": int(len(labels)),
    }

    out_dir = exports_dir
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    json_path = out_dir / f"{model_name}_ytf_roc_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

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
        "--model", required=True, help="facenet / adaface / arcface ..."
    )
    parser.add_argument(
        "--stamp", required=True, help="timestamp part, e.g. 20251110-182757"
    )
    parser.add_argument(
        "--exports-dir", default=None, help="override exports dir (optional)"
    )
    args = parser.parse_args()

    run_roc_ytf(args.model, args.stamp, args.exports_dir)
