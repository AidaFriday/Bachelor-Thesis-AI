import os
import sys
from pathlib import Path
from datetime import datetime
import json

import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

# make project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from connector import load_model  # noqa: E402

# re-use helpers from your existing video logic
try:
    # When run as part of the package
    from .logic_confusion_matrix_video import cosine_similarity, _resolve_video_root
except ImportError:
    # When run as a top-level script from project root
    from benchmark_parameters.validation_accuracy.video.logic_confusion_matrix_video import (  # type: ignore  # noqa: E501
        cosine_similarity,
        _resolve_video_root,
    )


def get_video_embedding(wrapper, video_dir: str, max_frames: int = 20):
    """
    Aggregate one embedding per video by averaging embeddings
    over up to `max_frames` frames.
    """
    if not os.path.isdir(video_dir):
        return None

    frame_files = sorted(
        f
        for f in os.listdir(video_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not frame_files:
        return None

    frame_files = frame_files[:max_frames]

    embs = []
    for fname in frame_files:
        img_path = os.path.join(video_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = wrapper.detector.detect(img)
        if not faces:
            continue

        aligned = wrapper.detector.align_for(img, faces[0]["kps"])
        if aligned is None:
            continue

        emb = wrapper.embed(aligned)
        if emb is not None:
            embs.append(emb)

    if not embs:
        return None

    return np.mean(embs, axis=0)


def load_ytf_meta(meta_path: str):
    """
    Load YTF meta_and_splits.mat and return (video_names, splits).
    """
    meta = loadmat(str(meta_path), squeeze_me=True)
    video_names = meta["video_names"]  # (3425,)
    splits = meta["Splits"]  # (500, 3, 10)
    return video_names, splits


def compute_ytf_pairs(
    wrapper,
    dataset_path: str,
    video_names,
    splits,
    fold_idx: int,
    max_frames_per_video: int = 100,
):
    """
    Compute similarities for the official YTF 500 video pairs in one fold.

    wrapper: loaded model (from connector.load_model)
    dataset_path: path to your YTF root (e.g. C:\\programming\\Datasets\\YTF)
    video_names: array from meta["video_names"]
    splits: array from meta["Splits"]
    fold_idx: which fold to use (0..9)
    max_frames_per_video: how many frames to sample per video directory
    """
    # normalize root so it works with both layouts:
    #   root/person/video/frame.jpg
    #   root/aligned_images_DB/person/video/frame.jpg
    root = _resolve_video_root(dataset_path)

    fold = splits[:, :, fold_idx]  # shape (500, 3)
    scores = []
    labels = []
    pair_records = []

    for idx1, idx2, is_same in tqdm(fold, desc=f"[YTF] Fold {fold_idx}"):
        idx1 = int(idx1) - 1  # meta is 1-based
        idx2 = int(idx2) - 1

        # video_names entries look like "Sadie_Frost/1"
        video_rel1 = video_names[idx1]
        video_rel2 = video_names[idx2]

        video_dir1 = os.path.join(root, video_rel1)
        video_dir2 = os.path.join(root, video_rel2)

        emb1 = get_video_embedding(wrapper, video_dir1, max_frames_per_video)
        emb2 = get_video_embedding(wrapper, video_dir2, max_frames_per_video)

        label = int(is_same)
        labels.append(label)

        error = (emb1 is None) or (emb2 is None)

        if error:
            # force always-wrong scores so these pairs are counted as failures
            score = -1.0 if label == 1 else 2.0
        else:
            score = cosine_similarity(emb1, emb2)

        scores.append(score)

        pair_records.append(
            {
                "video1": str(video_rel1),
                "video2": str(video_rel2),
                "label": "same" if label == 1 else "diff",
                "similarity": float(score),
                "error": bool(error),
            }
        )

    return (
        np.array(scores, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        pair_records,
    )


def export_ytf_pairs(model_name: str, ytf_root: str, meta_path: str, fold: int | None):
    """
    Export YTF pairs for one fold or all folds.

    Saves:
      - <model>_ytf_foldX_<timestamp>_scores.npy
      - <model>_ytf_foldX_<timestamp>_labels.npy
      - <model>_ytf_foldX_<timestamp>_pairs.json
    """
    print(f"[YTF] Model:   {model_name}")
    print(f"[YTF] Dataset: {ytf_root}")
    print(f"[YTF] Meta:    {meta_path}")

    wrapper = load_model(model_name)
    video_names, splits = load_ytf_meta(meta_path)

    n_folds = splits.shape[2]

    if fold is None:
        fold_list = list(range(n_folds))
    else:
        if not (0 <= fold < n_folds):
            raise ValueError(f"Fold must be in [0,{n_folds-1}], got {fold}")
        fold_list = [fold]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    export_dir = Path(__file__).resolve().parents[2] / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    for f_idx in fold_list:
        print(f"\n[YTF] ==== Fold {f_idx} ====")
        scores, labels, pair_records = compute_ytf_pairs(
            wrapper,
            ytf_root,
            video_names,
            splits,
            fold_idx=f_idx,
            max_frames_per_video=100,
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
    parser.add_argument(
        "--fold",
        type=int,
        default=-1,
        help="-1 = all folds, otherwise 0..9",
    )
    args = parser.parse_args()

    fold_arg = None if args.fold < 0 else args.fold
    export_ytf_pairs(args.model, args.dataset, args.meta, fold_arg)
