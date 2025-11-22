# logic_confusion_ytf_pairs.py
# Computes confusion metrics for YTF with auto-detected export folder

import os
import sys
import json
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

FIXED_THRESHOLD = 0.60


def find_ytf_export_dir(model_name: str) -> Path:
    """Auto-detect the YTF folder that contains <model>_ytf_fold0_* files."""
    if sys.platform.startswith("linux"):
        base = Path("/home/aida/github/BA_Utilites/BA_tests/Test_YTF")
    else:
        base = Path("C:/programming/BA_Utilites/BA_tests/Test_YTF")

    if not base.exists():
        raise FileNotFoundError(f"[ERROR] Test_YTF folder not found: {base}")

    # List all YTF_Video_* folders
    ytf_dirs = [
        d
        for d in base.iterdir()
        if d.is_dir() and d.name.lower().startswith("ytf_video_")
    ]

    model = model_name.lower()
    matched_dirs = []

    # Look inside each /folds for model-specific files
    for d in ytf_dirs:
        folds_path = d / "folds"
        if not folds_path.exists():
            continue

        pattern = f"{model}_ytf_fold0_"
        has_model_data = any(
            f.name.lower().startswith(pattern)
            for f in folds_path.iterdir()
            if f.is_file()
        )

        if has_model_data:
            matched_dirs.append(d)

    if matched_dirs:
        newest = max(matched_dirs, key=lambda x: x.stat().st_mtime)
        print(f"[AUTO] Found YTF results for model '{model_name}' → {newest}")
        return newest / "folds"

    # fallback
    newest = max(ytf_dirs, key=lambda x: x.stat().st_mtime)
    print(f"[AUTO] WARNING: No model folder found. Using → {newest}")
    return newest / "folds"


def load_all_scores_labels(exports_dir: Path, model: str, stamp: str):
    scores_list = []
    labels_list = []

    for fold in range(10):
        prefix = f"{model}_ytf_fold{fold}_{stamp}"
        scores_path = exports_dir / f"{prefix}_scores.npy"
        labels_path = exports_dir / f"{prefix}_labels.npy"

        if not scores_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"Missing files: {scores_path}, {labels_path}")

        scores_list.append(np.load(scores_path))
        labels_list.append(np.load(labels_path))

    return np.concatenate(scores_list), np.concatenate(labels_list)


def run_confusion_ytf(model_name: str, stamp: str):
    exports_dir = find_ytf_export_dir(model_name)

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

    # use ACTUAL timestamp for output
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # SAVE CONFUSION MATRIX PNG
    cm = np.array([[tp, fp], [fn, tn]])

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Actual +", "Actual -"])
    ax.set_yticklabels(["Predicted +", "Predicted -"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", color="black")

    ax.set_title(f"YTF Confusion Matrix – {model_name} (thr={FIXED_THRESHOLD})")

    png_path = exports_dir / f"{model_name}_ytf_confusion_{ts}.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()

    # SAVE JSON
    out_json = exports_dir / f"{model_name}_ytf_confusion_{ts}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[YTF CONF] Saved JSON → {out_json}")
    print(f"[YTF CONF] Saved PNG  → {png_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (e.g. arcface)")
    parser.add_argument("--stamp", required=True, help="Timestamp like 20251120-204342")
    args = parser.parse_args()

    run_confusion_ytf(args.model, args.stamp)
