# ==== performance_gpu/fps_gpu.py ====

import os
import sys

# -------------------------------------------------------------
# Resolve project src path   (.../Bachelor-Thesis-AI/src)
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

PROJECT_ROOT = os.path.dirname(SRC_DIR)

# -------------------------------------------------------------
# Standard imports
# -------------------------------------------------------------
import argparse
import json
import time
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
# High-accuracy GPU benchmark (STREAMING)
# -------------------------------------------------------------
def run(model_name, dataset_path, iters, frame_size):

    print(">>> gpu_fps started", flush=True)

    if not os.path.exists(dataset_path):
        print(json.dumps({"error": "Dataset path invalid"}))
        sys.exit(1)

    # iters == 0 → full dataset
    if iters <= 0:
        iters = None

    paths = YTF.list_all_images(root_dir=dataset_path, shuffle=False, verbose=False)
    if len(paths) == 0:
        print(json.dumps({"error": "Dataset contains no frames"}))
        sys.exit(1)

    if iters is not None:
        paths = paths[:iters]

    num_frames = len(paths)
    print(f"Loaded {num_frames} image paths from YTF", flush=True)

    # ---------- Load detector + model on CUDA ----------
    print(">>> Initializing models on GPU", flush=True)
    detector = FaceDetectorAligner(device="cuda")
    embedder = load_model(model_name, device="cuda")

    # ---------- Warm-up ----------
    warmup_iters = 30
    print(f"🔥 Warm-up on GPU ({warmup_iters} iterations)...", flush=True)

    first_img = cv2.imread(paths[0])
    for _ in range(warmup_iters):
        cuda_sync()
        process_frame(detector, embedder, first_img)
        cuda_sync()

    # ---------- Benchmark ----------
    print("🚀 Running GPU FPS benchmark...", flush=True)
    run_start = datetime.now().isoformat(timespec="seconds")

    cuda_sync()
    t0 = time.perf_counter()

    for i, p in enumerate(paths):
        frame = cv2.imread(p)
        if frame is None:
            continue

        process_frame(detector, embedder, frame)

        if (i + 1) % 50 == 0 or (i + 1 == num_frames):
            print(f"Progress: {i+1}/{num_frames}", flush=True)

    cuda_sync()
    t1 = time.perf_counter()

    total_time = t1 - t0
    fps = num_frames / total_time

    run_end = datetime.now().isoformat(timespec="seconds")

    result = {
        "kind": "gpu_fps",
        "model": model_name,
        "dataset": "YTF",
        "iters": num_frames,
        "start_time": run_start,
        "end_time": run_end,
        "total_time_sec": round(total_time, 4),
        "fps": round(fps, 2),
    }

    out_file = os.path.join(PROJECT_ROOT, "fps_gpu_report.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=4)

    print("=" * 40)
    print(f"🔥 GPU FPS: {fps:.2f}")
    print(f"📄 Saved → {out_file}")
    print("=" * 40)
    print(json.dumps(result), flush=True)


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--frame-size", type=str, default="640x640")

    args = parser.parse_args()
    run(args.model, args.dataset, args.iters, args.frame_size)


if __name__ == "__main__":
    main()
