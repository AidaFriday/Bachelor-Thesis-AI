# ==== performance_gpu/fps_gpu.py ====

import os
import sys

# -------------------------------------------------------------
# Resolve project src path   (.../Bachelor-Thesis-AI/src)
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))                      # /src/benchmark_parameters/performance_gpu
SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))     # /src

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Project root (used only for saving JSON)
PROJECT_ROOT = os.path.dirname(SRC_DIR)                                      # /Bachelor-Thesis-AI


# -------------------------------------------------------------
# Standard imports
# -------------------------------------------------------------
import argparse
import json
import time
import numpy as np
import cv2
from datetime import datetime

from src.connector import load_model
from models.wrap_facedetection import FaceDetectorAligner
from dataset import YTF

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# -------------------------------------------------------------
# GPU sync utility
# -------------------------------------------------------------
def cuda_sync():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.synchronize()


# -------------------------------------------------------------
# Full detect → align → embed pipeline
# -------------------------------------------------------------
def process_frame(detector, embedder, frame):
    dets = detector.detect(frame)
    if not dets:
        return None

    best = max(dets, key=lambda d: d["conf"])
    aligned = detector.align_for(frame, best["kps"], out_size=(160, 160))
    if aligned is None:
        return None

    if hasattr(embedder, "embed"):
        return embedder.embed(aligned)
    if hasattr(embedder, "get_embedding"):
        return embedder.get_embedding(aligned)

    raise RuntimeError("Embedder has no embed() or get_embedding()!")


# -------------------------------------------------------------
# Load YTF frames
# -------------------------------------------------------------
def load_ytf_frames(root, limit=None):
    paths = YTF.list_all_images(root_dir=root, shuffle=False, verbose=False)
    frames = []

    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            frames.append(img)
        if limit and len(frames) >= limit:
            break

    return frames, paths


# -------------------------------------------------------------
# High-accuracy GPU benchmark
# -------------------------------------------------------------
def run(model_name, dataset_path, iters, frame_size):

    if not os.path.exists(dataset_path):
        print(json.dumps({"error": "Dataset path invalid"}))
        sys.exit(1)

    frames, paths = load_ytf_frames(dataset_path, limit=iters)
    if len(frames) == 0:
        print(json.dumps({"error": "Dataset contains no frames"}))
        sys.exit(1)

    print(f"Loaded {len(frames)} frames from YTF")

    # ---------- Load detector + model on CUDA ----------
    detector = FaceDetectorAligner(device="cuda")
    embedder = load_model(model_name, device="cuda")

    # ---------- Warm-up (stabilizes GPU) ----------
    warmup_iters = 30
    print(f"🔥 Warm-up on GPU ({warmup_iters} iterations)...")

    test_frame = frames[0]
    for _ in range(warmup_iters):
        cuda_sync()
        process_frame(detector, embedder, test_frame)
        cuda_sync()

    # ---------- Benchmark ----------
    print("🚀 Running GPU FPS benchmark...")
    run_start = datetime.now().isoformat(timespec="seconds")

    cuda_sync()
    t0 = time.perf_counter()

    for i in range(iters):
        frame = frames[i % len(frames)]
        process_frame(detector, embedder, frame)

        if (i + 1) % 20 == 0 or (i + 1 == iters):
            print(f"Progress: {i+1}/{iters}")

    cuda_sync()
    t1 = time.perf_counter()

    total_time = t1 - t0
    fps = iters / total_time

    run_end = datetime.now().isoformat(timespec="seconds")

    result = {
        "kind": "gpu_fps",
        "model": model_name,
        "dataset": "YTF",
        "iters": iters,
        "start_time": run_start,
        "end_time": run_end,
        "total_time_sec": round(total_time, 4),
        "fps": round(fps, 2),
    }

    # Save JSON
    out_file = os.path.join(PROJECT_ROOT, "fps_gpu_report.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=4)

    print("=" * 40)
    print(f"🔥 GPU FPS: {fps:.2f}")
    print(f"📄 Saved → {out_file}")
    print("=" * 40)

    print(json.dumps(result))


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--frame-size", type=str, default="640x640")

    args = parser.parse_args()
    run(args.model, args.dataset, args.iters, args.frame_size)


if __name__ == "__main__":
    main()
