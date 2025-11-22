# updated script logic_roc_ytf_pairs.py with the location in BA_Utilities
# logic_roc_ytf_pairs.py
# Auto-detects YTF fold output folders on Windows or Linux.

# run_roc_ytf_pairs.py
# Auto-detects the correct model-specific YTF fold directory.

import os
import sys
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


def find_ytf_export_dir(model_name: str) -> Path:
    """
    AUTO-DETECT the correct YTF fold directory for a specific model.

    NEW LOGIC:
      1. Scan all YTF_Video_* folders.
      2. Inside each folder, look for files like:
         <model>_ytf_fold0_*.npy
      3. Choose the newest folder that actually contains this model.
      4. Fallback to generic only if absolutely necessary.
    """

    if sys.platform.startswith("linux"):
        base = Path("/home/aida/github/BA_Utilites/BA_tests/Test_YTF")
    else:
        base = Path("C:/programming/BA_Utilites/BA_tests/Test_YTF")

    if not base.exists():
        raise FileNotFoundError(f"[ERROR] Test_YTF folder not found: {base}")

    # find all candidate YTF_Video_* dirs
    ytf_dirs = [
        d
        for d in base.iterdir()
        if d.is_dir() and d.name.lower().startswith("ytf_video_")
    ]

    if not ytf_dirs:
        raise FileNotFoundError("[ERROR] No YTF_Video_* folders found.")

    model = model_name.lower()

    # STEP 1: search inside each YTF folder for model-matching files
    matched_dirs = []
    for d in ytf_dirs:
        folds_path = d / "folds"
        if not folds_path.exists():
            continue

        # look for files like: model_ytf_fold0_*.npy
        pattern = f"{model}_ytf_fold0_"
        has_model_data = any(
            f.name.lower().startswith(pattern)
            for f in folds_path.iterdir()
            if f.is_file()
        )

        if has_model_data:
            matched_dirs.append(d)

    # If we found model-specific dirs → choose the newest
    if matched_dirs:
        newest = max(matched_dirs, key=lambda x: x.stat().st_mtime)
        print(f"[AUTO] Found YTF results for model '{model_name}' → {newest}")
        return newest / "folds"

    # STEP 2: fallback — use newest generic folder
    newest_generic = max(ytf_dirs, key=lambda x: x.stat().st_mtime)
    print(
        f"[AUTO] WARNING: No model-specific files found. Using generic → {newest_generic}"
    )
    return newest_generic / "folds"


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


def run_roc_ytf(model_name: str, stamp: str):
    exports_dir = find_ytf_export_dir(model_name)

    scores, labels = load_all_scores_labels(exports_dir, model_name, stamp)

    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    print(f"[YTF ROC] pos_pairs={pos}, neg_pairs={neg}")

    fpr, tpr, thr = roc_curve(labels, scores)
    auc_val = auc(fpr, tpr)

    # best threshold
    best_idx = np.argmax(tpr - fpr)
    best_thr = float(thr[best_idx])

    # EER
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fnr[eer_idx] + fpr[eer_idx]) / 2)

    # TAR@FAR 1e-3
    target_far = 1e-3
    idx = np.searchsorted(fpr, target_far, side="right") - 1
    tar_far_1e3 = float(tpr[idx]) if 0 <= idx < len(tpr) else float("nan")

    metrics = {
        "kind": "roc_ytf",
        "model": model_name,
        "dataset": "YTF",
        "auc": float(auc_val),
        "eer": float(eer),
        "best_threshold": best_thr,
        "tar_far_1e3": tar_far_1e3,
        "pairs": len(labels),
        "pos_pairs": pos,
        "neg_pairs": neg,
    }

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_json = exports_dir / f"{model_name}_ytf_roc_{ts}.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[YTF ROC] JSON saved -> {out_json}")
    print(json.dumps(metrics, indent=2))

    # Save ROC curve PNG
    png_path = out_json.with_suffix(".png")  # saves .png in folds folder
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"YTF ROC – {model_name}")
    plt.legend(loc="lower right")
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"[YTF ROC] PNG saved -> {png_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--stamp", required=True)
    args = p.parse_args()

    run_roc_ytf(args.model, args.stamp)
