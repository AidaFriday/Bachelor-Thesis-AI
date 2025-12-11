# ytf_precompute_embeddings.py
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm
from scipy.io import loadmat

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from connector import load_model


# Detect if dataset is already aligned
def dataset_is_aligned(dataset_path: str) -> bool:
    p = dataset_path.lower().replace("\\", "/")
    last_dir = p.rstrip("/").split("/")[-1]

    if "aligned" in p:
        return True
    if last_dir in ("ytf", "aligned"):
        return True
    if "lfw" in p and "deepfunneled" in p:
        return True

    return False


# Uniform frame sampling for videos
def sample_uniform(files, max_frames):
    if len(files) <= max_frames:
        return files
    idx = (
        np.linspace(0, len(files) - 1, max_frames).round().astype(int)
    )  # it takes frames evenly spaced across the whole video, does not take the first consecutive frames, does not take random frames
    return [files[i] for i in idx]


# Load YTF metadata, meta_and_splits.mat file contains the list of all video folder names, the official 10 evaluation folds, returns the list of video names
def load_ytf_meta(meta_path: str):
    meta = loadmat(str(meta_path), squeeze_me=True)
    return meta["video_names"]


# Main embedding extractor


def get_video_embedding(
    wrapper, video_dir, max_frames, USE_DETECTION
):  # extract a single video embedding
    if not os.path.isdir(video_dir):
        return None

    frame_files = sorted(
        f
        for f in os.listdir(video_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not frame_files:
        return None

    frame_files = sample_uniform(frame_files, max_frames)  # Load available frame images

    embs = []
    model_name = getattr(wrapper, "name", "").lower()

    # Identify which model wrapper is being used
    is_arcface = model_name == "arcface"
    is_adaface = model_name == "adaface"
    is_facenet = model_name in ("facenet", "facenet_onnx")

    for fname in frame_files:
        img_path = os.path.join(video_dir, fname)

        # ArcFace full internal pipeline, includes face detection alignment, embedding extraction

        if is_arcface:
            emb = wrapper.get_embedding(img_path)
            if emb is None:
                continue

            emb = emb.astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-6
            embs.append(emb)
            continue

        # load image
        img = cv2.imread(img_path)
        if img is None:
            continue

        emb = None

        try:
            # Facenet - evaluated on 160×160 crops that are aligned using the shared 5-point alignment pipeline, even when the dataset already provides pre-aligned faces

            if is_facenet:
                faces = wrapper.detector.detect(img)
                if not faces:
                    continue

                aligned = wrapper.detector.align_for(img, faces[0]["kps"])
                if aligned is None:
                    continue

                emb = wrapper.embed(
                    aligned
                )  # takes an already aligned, correctly sized face crop (e.g. 160×160 for FaceNet), and runs the model’s forward pass to produce a feature embedding

            # AdaFace - direct embedding (YTF already aligned)
            elif is_adaface:
                emb = wrapper.embed(img)

            # other models - optional detect + align

            elif USE_DETECTION:
                faces = wrapper.detector.detect(img)
                if not faces:
                    continue

                aligned = wrapper.detector.align_for(img, faces[0]["kps"])
                if aligned is None:
                    continue

                emb = wrapper.embed(aligned)

            else:
                emb = wrapper.embed(img)

        except Exception:
            emb = None

        if emb is None:
            continue

        # -------------------------------------------------
        # Per-frame L2 normalization
        # -------------------------------------------------
        emb = emb.astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-6
        embs.append(emb)

    if not embs:
        return None

    # ---------------------------------------------------------
    # Mean pooling across frames
    # ---------------------------------------------------------
    v = np.mean(embs, axis=0)
    v /= np.linalg.norm(v) + 1e-6
    return v.astype(np.float32)


# ---------------------------------------------------------
# Main YTF loop
# ---------------------------------------------------------
def precompute_ytf_embeddings(
    model_name: str, ytf_root: str, meta_path: str, max_frames: int = 10
):
    print("─────────────────────────────────────────────")
    print(f"[YTF] Model:        {model_name}")
    print(f"[YTF] Dataset root: {ytf_root}")
    print(f"[YTF] Meta path:    {meta_path}")
    print(f"[YTF] max_frames:   {max_frames}")
    print("─────────────────────────────────────────────")

    wrapper = load_model(model_name)

    # -------------------------------
    # GPU support for PyTorch models
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using {device}")

    if device.type == "cuda" and hasattr(wrapper, "model"):
        try:
            wrapper.model = wrapper.model.to(device)
        except Exception:
            print("[WARN] Could not move model to CUDA")

    # load meta
    names = load_ytf_meta(meta_path)

    # detect YTF aligned folder
    if os.path.isdir(os.path.join(ytf_root, "aligned")):
        video_root = os.path.join(ytf_root, "aligned")
    else:
        video_root = ytf_root

    USE_DETECTION = not dataset_is_aligned(video_root)
    print(f"[INFO] USE_DETECTION = {USE_DETECTION}\n")

    all_embs = []
    all_names = []
    failed = []
    total_start = time.time()

    for name in tqdm(names, desc="[YTF] Videos", ncols=100):
        name_str = str(name)
        vid_dir = os.path.join(video_root, name_str)

        t0 = time.time()
        emb = get_video_embedding(wrapper, vid_dir, max_frames, USE_DETECTION)
        dur = time.time() - t0

        if emb is None:
            print(f"[SKIP] {name_str:40s} ({dur:5.2f}s) no usable frames")
            failed.append(name_str)
        else:
            print(f"[OK]   {name_str:40s} ({dur:5.2f}s)")
            all_names.append(name_str)
            all_embs.append(emb)

    if not all_embs:
        print("[ERROR] No embeddings computed.")
        return None

    # ---------------------------------------------------------
    # CUSTOM LINUX OUTPUT DIRECTORY
    # ---------------------------------------------------------
    if sys.platform.startswith("linux"):
        base_root = Path("/home/aida/github/BA_Utilites/BA_tests/Test_YTF")
    else:
        base_root = Path(__file__).resolve().parents[2] / "exports"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_dir = base_root / f"YTF_Video_{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=True)

    base = f"{model_name}_ytf_video_embs"

    npz_path = export_dir / f"{base}.npz"
    json_path = export_dir / f"{base}.json"

    np.savez_compressed(
        npz_path, names=np.array(all_names), embs=np.stack(all_embs, axis=0)
    )

    summary = {
        "model": model_name,
        "dataset": "YTF",
        "max_frames": max_frames,
        "num_success": len(all_names),
        "num_failed": len(failed),
        "failed": failed,
        "runtime_min": round((time.time() - total_start) / 60, 2),
        "timestamp": timestamp,
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n─────────────────────────────────────────────")
    print(f"[YTF] Saved embeddings: {npz_path}")
    print(f"[YTF] Saved summary:    {json_path}")
    print(f"[YTF] Total runtime:    {summary['runtime_min']} min")
    print("─────────────────────────────────────────────\n")

    return str(npz_path)


# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--max-frames", type=int, default=10)
    args = p.parse_args()

    precompute_ytf_embeddings(
        args.model,
        args.dataset,
        args.meta,
        max_frames=args.max_frames,
    )
