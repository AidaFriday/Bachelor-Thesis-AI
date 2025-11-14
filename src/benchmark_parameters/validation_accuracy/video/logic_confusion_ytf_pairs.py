# Computes confusion matrix and evaluation metrics for YTF (all 10 folds)
# using precomputed scores and labels, and exports JSON + confusion matrix PNG.

import os
import sys
import json
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

FIXED_THRESHOLD = 0.60

def load_all_scores_labels(exports_dir: Path, model: str, stamp: str):
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


def run_confusion_ytf(model_name: str, stamp: str, exports_dir: str | None = None):
    if exports_dir is None:
        exports_dir = Path(__file__).resolve().parents[2] / "exports"
    else:
        exports_dir = Path(exports_dir)

    scores, labels = load_all_scores_labels(exports_dir, model_name, stamp)

    preds = (scores >= FIXED_THRESHOLD).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    far = fp / (fp + tn + 1e-9)
    frr = fn / (fn + tp + 1e-9)

    result = {
        "kind": "confusion_matrix_ytf",
        "model": model_name,
        "dataset": "YTF",
        "pairs": int(len(labels)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "far": float(far),
        "frr": float(frr),
        "threshold": float(FIXED_THRESHOLD),
    }

    # ---------- NEW: draw confusion-matrix PNG ----------
    cm = np.array([[tp, fp], [fn, tn]])

    labels_cm = [["TP", "FP"], ["FN", "TN"]]

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Actual +", "Actual -"])
    ax.set_yticklabels(["Predicted +", "Predicted -"])

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{labels_cm[i][j]}\n{cm[i, j]}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    ax.set_title(f"YTF Confusion Matrix – {model_name} (thr={FIXED_THRESHOLD})")
    fig.tight_layout()

    png_path = Path(__file__).with_name("confusion_ytf_result.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    # ---------- END NEW PART ----------

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = exports_dir / f"{model_name}_ytf_confusion_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[YTF CM] Exported -> {out_path}")
    print(json.dumps(result))

    # optional extra JSON line so GUI can pick up the PNG
    print(
        json.dumps(
            {
                "kind": "confusion_image_ytf",
                "path": str(png_path),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "threshold": float(FIXED_THRESHOLD),
                "model": model_name,
                "dataset": "YTF",
            }
        )
    )


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

    run_confusion_ytf(args.model, args.stamp, args.exports_dir)
