# Computes one embedding per YTF video by averaging multiple frames,
# using FACENET **CPU ONLY** (because GPU is unsupported on RTX-5070 for PyTorch)

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import torch

# ── FORCE CPU ALWAYS ───────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""     # disables GPU globally
torch.backends.cudnn.enabled = False        # no GPU kernels
torch.set_num_threads(4)                    # optional: avoid CPU overload

print("\n[INFO] Running FACENET strictly on CPU.\n")

# ── make project root importable ───────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from connector import load_model  # noqa: E402

try:
    from .logic_confusion_matrix_video import _resolve_video_root
except ImportError:
    from benchmark_parameters.validation_accuracy.video.logic_confusion_matrix_video import (
        _resolve_video_root,
    )


# ── metadata loader ────────────────────────────────────────────────────────────
def load_ytf_meta(meta_path: str):
    meta = loadmat(str(meta_path), squeeze_me=True)
    return meta["video_names"]


# ── per-video embedding ────────────────────────────────────────────────────────
def get_video_embedding(wrapper, video_dir: str, max_frames: int = 10):
    if not os.path.isdir(video_dir):
        return None

    frame_files = sorted(
        f for f in os.listdir(video_dir)
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

        # IMPORTANT: CPU-ONLY embedding call
        emb = wrapper.embed(aligned)

        if emb is not None:
            embs.append(emb)

    if not embs:
        return None

    return np.mean(embs, axis=0)


# ── main driver ────────────────────────────────────────────────────────────────
def precompute_ytf_embeddings(model_name: str, ytf_root: str, meta_path: str, max_frames: int = 10):
    print("─────────────────────────────────────────────")
    print(f"[YTF-PRECOMPUTE] Model:   {model_name}")
    print(f"[YTF-PRECOMPUTE] Dataset: {ytf_root}")
    print(f"[YTF-PRECOMPUTE] Meta:    {meta_path}")
    print(f"[YTF-PRECOMPUTE] max_frames_per_video = {max_frames}")
    print("─────────────────────────────────────────────")

    print("[CPU] Using CPU for ALL operations.\n")

    # always CPU
    device = torch.device("cpu")

    wrapper = load_model(model_name)
    wrapper.device = "cpu"      # enforce CPU inside wrapper
    if hasattr(wrapper, "model"):
        wrapper.model = wrapper.model.to("cpu")

    video_names = load_ytf_meta(meta_path)
    root = _resolve_video_root(ytf_root)

    names_list, emb_list, failed = [], [], []
    total_start = time.time()

    for i, name in enumerate(tqdm(video_names, desc="[YTF] videos", ncols=100)):
        name_str = str(name)
        video_dir = os.path.join(root, name_str)
        t0 = time.time()

        emb = get_video_embedding(wrapper, video_dir, max_frames=max_frames)
        dur = time.time() - t0

        if emb is None:
            print(f"[SKIP] {name_str:40s}  ({dur:6.2f}s)  no face")
            failed.append(name_str)
        else:
            print(f"[OK]   {name_str:40s}  ({dur:6.2f}s)")
            names_list.append(name_str)
            emb_list.append(emb.astype(np.float32))

        if (i + 1) % 50 == 0:
            elapsed = (time.time() - total_start) / 60
            print(f"[PROGRESS] {i+1}/{len(video_names)} done, elapsed {elapsed:.1f} min")

    if not emb_list:
        print("[YTF-PRECOMPUTE] No embeddings computed, aborting.")
        return None

    names_arr = np.array(names_list)
    embs_arr = np.stack(emb_list, axis=0)

    export_dir = Path(__file__).resolve().parents[2] / "exports"
    export_dir.mkdir(exist_ok=True, parents=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{model_name}_ytf_video_embs_cpu_maxf{max_frames}_{ts}"
    npz_path = export_dir / f"{base}.npz"
    json_path = export_dir / f"{base}.json"

    np.savez_compressed(npz_path, names=names_arr, embs=embs_arr)
    runtime_min = round((time.time() - total_start) / 60, 2)

    summary = {
        "model": model_name,
        "dataset": "YTF",
        "max_frames": max_frames,
        "num_success": int(len(names_arr)),
        "num_failed": int(len(failed)),
        "failed_videos": failed,
        "runtime_min": runtime_min,
        "timestamp": ts,
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n─────────────────────────────────────────────")
    print(f"[YTF-PRECOMPUTE] Saved embeddings → {npz_path}")
    print(f"[YTF-PRECOMPUTE] Summary JSON →     {json_path}")
    print(f"[YTF-PRECOMPUTE] Total time: {runtime_min} min")
    print("─────────────────────────────────────────────\n")

    return str(npz_path)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--max-frames", type=int, default=10)
    args = parser.parse_args()

    precompute_ytf_embeddings(args.model, args.dataset, args.meta, max_frames=args.max_frames)
