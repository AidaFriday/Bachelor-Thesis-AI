# Computes one embedding per YTF video by averaging multiple frames,
# then saves embeddings to .npz for fast reuse in evaluation.

import os, sys, time, json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
from scipy.io import loadmat
from tqdm import tqdm

# ── make project root importable ───────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from connector import load_model  # noqa: E402

try:
    from .logic_confusion_matrix_video import _resolve_video_root
except ImportError:
    from benchmark_parameters.validation_accuracy.video.logic_confusion_matrix_video import (  # type: ignore
        _resolve_video_root,
    )


# ── metadata loader ────────────────────────────────────────────────────────────
def load_ytf_meta(meta_path: str):
    """Load YTF meta_and_splits.mat and return video_names."""
    meta = loadmat(str(meta_path), squeeze_me=True)
    return meta["video_names"]  # (3425,)


# ── per-video embedding ────────────────────────────────────────────────────────
def get_video_embedding(wrapper, video_dir: str, max_frames: int = 10):
    """Aggregate one embedding per video by averaging embeddings of multiple frames."""
    if not os.path.isdir(video_dir):
        return None

    frame_files = sorted(
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not frame_files:
        return None

    # Evenly sample up to max_frames frames
    if len(frame_files) > max_frames:
        step = len(frame_files) // max_frames
        frame_files = frame_files[::step][:max_frames]

    aligned_faces = []
    for fname in frame_files:
        img_path = os.path.join(video_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = wrapper.detector.detect(img)
        if not faces:
            continue
        aligned = wrapper.detector.align_for(img, faces[0]["kps"])
        if aligned is not None:
            aligned_faces.append(aligned)

    if not aligned_faces:
        return None

    # batched GPU embedding
    embs = wrapper.embed_batch(aligned_faces, batch_size=32)
    if embs is None or len(embs) == 0:
        return None

    return np.mean(embs, axis=0)  # single embedding per video


# ── main driver ────────────────────────────────────────────────────────────────
def precompute_ytf_embeddings(model_name, ytf_root, meta_path, max_frames=10):
    print("─────────────────────────────────────────────")
    print(f"[YTF-PRECOMPUTE] Model:   {model_name}")
    print(f"[YTF-PRECOMPUTE] Dataset: {ytf_root}")
    print(f"[YTF-PRECOMPUTE] Meta:    {meta_path}")
    print(f"[YTF-PRECOMPUTE] max_frames_per_video = {max_frames}")
    print("─────────────────────────────────────────────")

    torch.backends.cudnn.benchmark = True  # enable cudnn autotuning

    wrapper = load_model(model_name)
    video_names = load_ytf_meta(meta_path)
    root = _resolve_video_root(ytf_root)

    # quick device + environment check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[GPU] Using CUDA device: {gpu_name}")
    else:
        print("[WARNING] CUDA not available, running on CPU.")

    names_list, emb_list, failed = [], [], []
    total_start = time.time()

    # helper
    def process_video(name):
        name_str = str(name)
        video_dir = os.path.join(root, name_str)
        t0 = time.time()
        emb = get_video_embedding(wrapper, video_dir, max_frames=max_frames)
        dur = time.time() - t0
        if emb is None:
            print(f"[SKIP] {name_str:40s}  ({dur:6.2f}s)  no face", flush=True)
        else:
            print(f"[OK]   {name_str:40s}  ({dur:6.2f}s)", flush=True)
        return name_str, emb

    max_threads = min(16, os.cpu_count() or 8)
    print(f"\n[YTF-PRECOMPUTE] Using {max_threads} loader threads\n")

    start = time.time()
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(process_video, n): n for n in video_names}
        for i, f in enumerate(tqdm(as_completed(futures), total=len(video_names), desc="[YTF] videos")):
            name_str, emb = f.result()
            if emb is None:
                failed.append(name_str)
                continue
            names_list.append(name_str)
            emb_list.append(emb.astype(np.float32))
            elapsed = (time.time() - start) / 60
            if (i + 1) % 50 == 0:  # show periodic timing info
                print(f"[PROGRESS] {i+1}/{len(video_names)} done, elapsed {elapsed:.1f} min", flush=True)

    if not emb_list:
        print("[YTF-PRECOMPUTE] No embeddings computed, aborting.")
        return None

    names_arr = np.array(names_list)
    embs_arr = np.stack(emb_list, axis=0)

    export_dir = Path(__file__).resolve().parents[2] / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{model_name}_ytf_video_embs_maxf{max_frames}_{ts}"
    npz_path = export_dir / f"{base}.npz"
    json_path = export_dir / f"{base}.json"

    np.savez_compressed(npz_path, names=names_arr, embs=embs_arr)
    summary = {
        "model": model_name,
        "dataset": "YTF",
        "max_frames": max_frames,
        "num_videos_meta": len(video_names),
        "num_success": len(names_arr),
        "num_failed": len(failed),
        "failed_videos": failed,
        "runtime_min": round((time.time() - total_start) / 60, 2),
        "timestamp": ts,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n─────────────────────────────────────────────")
    print(f"[YTF-PRECOMPUTE] Saved embeddings → {npz_path}")
    print(f"[YTF-PRECOMPUTE] Summary JSON →     {json_path}")
    print(f"[YTF-PRECOMPUTE] Total time: {summary['runtime_min']} min")
    print("─────────────────────────────────────────────\n")

    return str(npz_path)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, help="Path to YTF root")
    parser.add_argument("--meta", required=True, help="Path to meta_and_splits.mat")
    parser.add_argument("--max-frames", type=int, default=10,
                        help="Frames per video to average")
    args = parser.parse_args()

    precompute_ytf_embeddings(args.model, args.dataset, args.meta,
                              max_frames=args.max_frames)
