import os
import sys
from pathlib import Path
from datetime import datetime
import json

import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

# === FIX PACKAGE PATH ISSUE ===
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]  # <project>/src
sys.path.insert(0, str(project_root))
# =================================

# make project root importable (kept for safety)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from connector import load_model  # noqa: E402


# re-use helpers from your existing video logic
# re-use helpers from your existing video logic
try:
    # When run as part of the package
    from .legacy.logic_confusion_matrix_video import (
        cosine_similarity,
        _resolve_video_root,
    )
except ImportError:
    # When run as a top-level script from project root
    from benchmark_parameters.validation_accuracy.video.legacy.logic_confusion_matrix_video import (
        cosine_similarity,
        _resolve_video_root,
    )


def load_precomputed_embs(npz_path: str):
    """
    Load precomputed YTF video embeddings from npz.

    Returns a dict:
        { "Person_X/1": embedding_vector, ... }
    """
    data = np.load(npz_path)
    names = data["names"]  # array of strings
    embs = data["embs"]  # array (N, D)
    return {str(n): e for n, e in zip(names, embs)}


def load_ytf_meta(meta_path: str):
    """
    Load YTF meta_and_splits.mat and return (video_names, splits).
    """
    meta = loadmat(str(meta_path), squeeze_me=True)
    video_names = meta["video_names"]  # (3425,)
    splits = meta["Splits"]  # (500, 3, 10)
    return video_names, splits


def compute_ytf_pairs(
    video_embs: dict,
    video_names,
    splits,
    fold_idx: int,
):
    """
    Compute similarities for the official YTF 500 video pairs in one fold,
    using *precomputed* embeddings.

    video_embs: dict { "Person/clip": embedding_vector }
    video_names: array from meta["video_names"]
    splits: array from meta["Splits"]
    fold_idx: which fold to use (0..9)
    """
    fold = splits[:, :, fold_idx]  # shape (500, 3)
    scores = []
    labels = []
    pair_records = []

    for idx1, idx2, is_same in tqdm(fold, desc=f"[YTF] Fold {fold_idx}"):
        idx1 = int(idx1) - 1  # meta is 1-based
        idx2 = int(idx2) - 1
        label = int(is_same)

        # video_names entries look like "Sadie_Frost/1"
        video_key1 = str(video_names[idx1])
        video_key2 = str(video_names[idx2])

        emb1 = video_embs.get(video_key1)
        emb2 = video_embs.get(video_key2)

        labels.append(label)

        # Handle missing embeddings
        if emb1 is None or emb2 is None:
            score = -1.0 if label == 1 else 2.0
            misclassified = True
        else:
            score = cosine_similarity(emb1, emb2)

            # REAL misclassification logic using the ROC best threshold
            THRESHOLD = 0.39875417947769165

            if label == 1:  # same
                misclassified = score < THRESHOLD
            else:  # diff
                misclassified = score > THRESHOLD

        scores.append(score)

        pair_records.append(
            {
                "video1": video_key1,
                "video2": video_key2,
                "label": "same" if label == 1 else "diff",
                "similarity": float(score),
                "misclassified": bool(misclassified),
            }
        )

    return (
        np.array(scores, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        pair_records,
    )


def export_ytf_pairs(
    model_name: str,
    ytf_root: str,
    meta_path: str,
    embs_path: str,
    fold: int | None,
):
    """
    Export YTF pairs for one fold or all folds, using precomputed embeddings.

    Saves:
      - <model>_ytf_foldX_<timestamp>_scores.npy
      - <model>_ytf_foldX_<timestamp>_labels.npy
      - <model>_ytf_foldX_<timestamp>_pairs.json
    """
    print(f"[YTF] Model:   {model_name}")
    print(f"[YTF] Dataset: {ytf_root}")
    print(f"[YTF] Meta:    {meta_path}")
    print(f"[YTF] Embs:    {embs_path}")

    # we no longer need load_model() here – embeddings are precomputed
    video_names, splits = load_ytf_meta(meta_path)
    video_embs = load_precomputed_embs(embs_path)

    n_folds = splits.shape[2]

    if fold is None:
        fold_list = list(range(n_folds))
    else:
        if not (0 <= fold < n_folds):
            raise ValueError(f"Fold must be in [0,{n_folds-1}], got {fold}")
        fold_list = [fold]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    # if main_folds.py passes an output directory, use it
    if hasattr(export_ytf_pairs, "outdir") and export_ytf_pairs.outdir is not None:
        export_dir = Path(export_ytf_pairs.outdir)
    else:
        export_dir = Path(__file__).resolve().parents[2] / "exports"

    export_dir.mkdir(parents=True, exist_ok=True)

    for f_idx in fold_list:
        print(f"\n[YTF] ==== Fold {f_idx} ====")
        scores, labels, pair_records = compute_ytf_pairs(
            video_embs,
            video_names,
            splits,
            fold_idx=f_idx,
        )

        fold_tag = f"fold{f_idx}"
        base = f"{model_name}_ytf_{fold_tag}_{ts}"

        np.save(export_dir / f"{base}_scores.npy", scores)
        np.save(export_dir / f"{base}_labels.npy", labels)

        export_json = {
            "meta": {
                "model": model_name,
                "dataset": "YTF",
                "fold": fold_tag,
                "num_pairs": int(len(labels)),
                "timestamp": ts,
            },
            "pairs": pair_records,
        }
        with open(export_dir / f"{base}_pairs.json", "w") as f:
            json.dump(export_json, f, indent=2)

        print(f"[YTF] Exported fold {f_idx} to:")
        print(f"       {export_dir / (base + '_scores.npy')}")
        print(f"       {export_dir / (base + '_labels.npy')}")
        print(f"       {export_dir / (base + '_pairs.json')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, help="Path to YTF root")
    parser.add_argument("--meta", required=True, help="Path to meta_and_splits.mat")
    parser.add_argument("--embs", required=True, help="Path to precomputed YTF .npz")
    parser.add_argument(
        "--fold",
        type=int,
        default=-1,
        help="-1 = all folds, otherwise 0..9",
    )

    parser.add_argument(
        "--outdir",
        required=False,
        default=None,
        help="Optional output directory for saving fold results",
    )

    args = parser.parse_args()

    fold_arg = None if args.fold < 0 else args.fold
    # pass outdir to the function via attribute (minimal change)
    export_ytf_pairs.outdir = args.outdir

    export_ytf_pairs(
        args.model,
        args.dataset,
        args.meta,
        args.embs,
        fold_arg,
    )
