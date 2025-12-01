# ==== benchmark_parameters/performance/fps/fps.py ====

import argparse
import json
import os
import sys
import time
import numpy as np
import cv2
from datetime import datetime


def detect_and_embed(detector_aligner, embedder, frame):
    """
    Perform detect + align + embed using your existing classes.
    detector_aligner: FaceDetectorAligner instance
    embedder: model wrapper (FaceNetONNX or FaceNetOriginalWrapper)
    """
    # 1) Detect faces
    dets = detector_aligner.detect(frame)
    if not dets:
        return None  # no face

    # choose highest-confidence detection
    det = max(dets, key=lambda d: d["conf"])
    kps = det["kps"]

    # 2) Align face
    aligned = detector_aligner.align_for(frame, kps, out_size=(160, 160))
    if aligned is None:
        return None

    # 3) Embed
    if hasattr(embedder, "embed"):
        return embedder.embed(aligned)
    elif hasattr(embedder, "get_embedding"):
        return embedder.get_embedding(aligned)
    else:
        raise RuntimeError("Model wrapper has no embed() or get_embedding().")


# ---- Bootstrap sys.path so connector and dataset are importable ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))  # src/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from connector import load_model
from dataset import YTF

try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")


# ---------------- Helpers ----------------
def send_log(msg, level="info"):
    payload = {"log": msg, "level": level}
    print(json.dumps(payload))
    sys.stdout.flush()


def _cuda_synchronize_if_needed():
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


def _random_frame(h=640, w=640):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def measure_once(detector, wrapper, frame):
    _cuda_synchronize_if_needed()
    t0 = time.perf_counter()
    _ = detect_and_embed(detector, wrapper, frame)
    _cuda_synchronize_if_needed()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def _ytf_loaded_subset_summary(root_dir: str, image_paths):
    """Summarize the loaded YTF subset (subjects/videos/frames)."""
    subjects, videos = set(), set()
    for p in image_paths:
        rel = os.path.relpath(p, root_dir)
        parts = rel.split(os.sep)
        if len(parts) >= 3:
            subjects.add(parts[0])
            videos.add(os.path.join(parts[0], parts[1]))
    return len(subjects), len(videos), len(image_paths)


# ---------------- Main Run ----------------
def run(model_name, iters, frame_h, frame_w, dataset):
    # --- Validate dataset (FPS must use YTF) ---
    if not dataset or not os.path.exists(dataset):
        send_log(
            "❌ Invalid dataset path. FPS benchmark requires YTF dataset.",
            level="error",
        )
        print(json.dumps({"error": "Invalid or missing YTF dataset path"}))
        sys.exit(1)

    dataset_lower = dataset.lower()
    if "lfw" in dataset_lower:
        send_log("❌ FPS test not available for LFW (image) datasets.", level="error")
        print(json.dumps({"error": "FPS benchmark supports only YTF (video) datasets"}))
        sys.exit(1)

    # Ensure aligned_images_DB subfolder is used
    aligned = os.path.join(dataset, "aligned_images_DB")
    if os.path.exists(aligned):
        dataset = aligned

    # ✅ benchmark start time
    run_start = datetime.now().isoformat(timespec="seconds")

    wrapper = load_model(model_name)
    send_log(f"Running FPS benchmark | Model: {model_name} | Dataset: YTF (video)")
    from models.wrap_facedetection import FaceDetectorAligner

    detector = FaceDetectorAligner(device="cpu")  # or "cuda"

    # --- Load YTF frames ---
    selected_subjects_env = os.getenv("YTF_SELECTED_SUBJECTS", "")
    selected_subjects = [
        s.strip() for s in selected_subjects_env.split(",") if s.strip()
    ]

    # First subject chosen in GUI (or None)
    start_identity = selected_subjects[0] if selected_subjects else None

    all_images = YTF.list_all_images(root_dir=dataset, shuffle=False, verbose=False)

    if selected_subjects:
        images = [
            p
            for p in all_images
            if os.path.basename(os.path.dirname(os.path.dirname(p)))
            in selected_subjects
        ]
        if not images:
            send_log("⚠️ No YTF frames found for selected subjects", level="error")
            print(json.dumps({"error": "No frames found for selected subjects"}))
            sys.exit(1)
        send_log(f"Filtering to selected subjects: {', '.join(selected_subjects)}")
    else:
        images = all_images

    if not images:
        send_log("⚠️ YTF dataset contains no images", level="error")
        print(json.dumps({"error": "Empty YTF dataset"}))
        sys.exit(1)

    s, v, f = _ytf_loaded_subset_summary(dataset, images)
    send_log(f"[YTF] Loaded subset: {s} subjects, {v} videos, {f} frames", level="info")
    frames = []
    image_map = {}
    if iters <= 0:
        iters = len(images)

    for idx, path in enumerate(images, 1):
        img = cv2.imread(path)
        if img is not None:
            frames.append(img)
            image_map[idx] = os.path.basename(path)
            if idx <= 3:
                send_log(f"Loaded image: {path}")

    # --- Multi-run support ---
    num_runs = int(os.getenv("YTF_RUNS", "1"))
    send_log(f"[CONFIG] Performing {num_runs} run(s) × {iters} frames each")

    # ---- Adaptive Warmup ----
    first_frame = None
    if len(images) > 0:
        first_frame = cv2.imread(images[0])
    if first_frame is None:
        send_log("❌ Could not read first frame for warm-up", "error")
        first_frame = _random_frame(frame_h, frame_w)

    # Decide warm-up count based on device
    if _HAS_TORCH and torch.cuda.is_available():
        warmup_iters = 5  # GPUs need more warm-up to stabilize
        device_name = "GPU"
    else:
        warmup_iters = 1  # CPU warm-up mostly negligible
        device_name = "CPU"

    send_log(f"🔥 Performing {warmup_iters} warm-up iteration(s) on {device_name}")

    for _ in range(warmup_iters):
        _ = detect_and_embed(detector, wrapper, first_frame)
        _cuda_synchronize_if_needed()

    # --- Initialize data collectors ---
    all_run_fps = []
    all_run_series = []
    used_filenames_all = []

    for run_idx in range(num_runs):
        send_log(f"--- Run {run_idx + 1}/{num_runs} ---")
        times_ms = []
        start = time.time()

        used_filenames = []

        for i in range(iters):
            frame_idx = i % len(frames) if frames else None
            frame = frames[frame_idx] if frames else _random_frame(frame_h, frame_w)
            t = measure_once(detector, wrapper, frame)

            times_ms.append(t)

            if frames:
                used_filenames.append(os.path.basename(images[frame_idx]))

            # ✅ Progress update every 10 frames or at end
            if (i + 1) % 10 == 0 or (i + 1) == iters:
                progress_msg = {
                    "_type": "progress",
                    "progress": i + 1,
                    "total": iters,
                    "run": run_idx + 1,
                    "num_runs": num_runs,
                }
                print(json.dumps(progress_msg), flush=True)

        elapsed = time.time() - start
        fps_series = [1000.0 / t if t > 0 else float("inf") for t in times_ms]
        # t = latency of one frame in milliseconds
        # 1000.0 / t → converts latency (ms) into FPS
        # FPS = “frames per second” = reciprocal of average per-frame processing time

        mean_fps = float(np.mean(fps_series))
        all_run_fps.append(mean_fps)
        all_run_series.append(fps_series)
        used_filenames_all.append(used_filenames)

        send_log(
            f"[Run {run_idx + 1}] {iters} frames → {mean_fps:.2f} FPS", level="result"
        )

    avg_fps = sum(all_run_fps) / len(all_run_fps)
    send_log(
        f"[RESULT] Average FPS over {num_runs} run(s): {avg_fps:.2f}", level="result"
    )

    # ✅ benchmark end time
    run_end = datetime.now().isoformat(timespec="seconds")

    payload = {
        "kind": "fps",
        "source_file": os.path.basename(__file__),
        "model": model_name,
        "dataset": "YTF (video)",
        "start_identity": start_identity,
        "start_time": run_start,
        "end_time": run_end,
        "avg_fps_all_runs": avg_fps,  # ✅ overall average
        "runs": all_run_fps,  # per-run averages
        "fps_series_all": all_run_series,
    }

    # --- Save per-run summary JSON ---
    report = {
        "source_file": os.path.basename(__file__),
        "model": model_name,
        "dataset": "YTF (video)",
        "start_identity": start_identity,
        "start_time": run_start,
        "end_time": run_end,
        "avg_fps_all_runs": round(avg_fps, 2),
        "runs": [],
    }

    for run_idx, fps_series in enumerate(all_run_series):
        min_idx = int(np.argmin(fps_series))
        max_idx = int(np.argmax(fps_series))

        used = used_filenames_all[run_idx] if run_idx < len(used_filenames_all) else []

        report["runs"].append(
            {
                "run": run_idx + 1,
                "min_fps": round(float(fps_series[min_idx]), 2),
                "max_fps": round(float(fps_series[max_idx]), 2),
                "avg_fps": round(all_run_fps[run_idx], 2),
                "min_file": (
                    used[min_idx]
                    if used and min_idx < len(used)
                    else f"frame_{min_idx + 1}"
                ),
                "max_file": (
                    used[max_idx]
                    if used and max_idx < len(used)
                    else f"frame_{max_idx + 1}"
                ),
            }
        )

    report_file = os.path.join(PROJECT_ROOT, "fps_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    send_log(f"Saved FPS summary report → {report_file}")

    print(json.dumps(payload))
    sys.stdout.flush()


# ---------------- CLI Entry ----------------
def _resolve_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--frame-size", type=str, default="640x640")
    parser.add_argument("--dataset", type=str, required=False)
    args = parser.parse_args()

    cfg = _resolve_settings()
    model = args.model or cfg.get("model")
    dataset = args.dataset or cfg.get("dataset")

    if not model:
        print(json.dumps({"error": "No model selected"}))
        sys.exit(1)

    try:
        h, w = [int(x) for x in args.frame_size.lower().split("x")]
    except Exception:
        h, w = 640, 640

    run(model, args.iters, h, w, dataset)


if __name__ == "__main__":
    main()
