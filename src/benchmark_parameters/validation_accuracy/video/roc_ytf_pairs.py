import os
import sys
from pathlib import Path
import json
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


def load_scores_labels(base_prefix):
    base = Path(base_prefix)
    scores = np.load(base.with_name(base.name + "_scores.npy"))
    labels = np.load(base.with_name(base.name + "_labels.npy"))
    return scores, labels


def run_roc_ytf(base_prefix, model_name="unknown"):
    scores, labels = load_scores_labels(base_prefix)

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

    export_dir = Path(__file__).resolve().parents[2] / "exports"
    export_dir.mkdir(exist_ok=True, parents=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    json_path = export_dir / f"{model_name}_ytf_roc_{ts}.json"
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
        "--base", required=True, help="Base prefix from logic_ytf_pairs"
    )
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    run_roc_ytf(args.base, args.model)
