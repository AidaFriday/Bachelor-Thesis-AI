# ==== logic_dataset_video.py ====
import json, sys, time, numpy as np, os, cv2
from datetime import datetime
from connector import load_model

try:
    import torch
    _HAS_TORCH = True
except:
    _HAS_TORCH = False


def print_progress_bar(percent, run, num_runs, width=40):
    filled = int(width * percent / 100)
    bar = "█" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {percent:3d}%  (Run {run}/{num_runs})")
    sys.stdout.flush()


def send_log(msg, level="info"):
    print(json.dumps({"log": msg, "level": level}), flush=True)


def _cuda_sync():
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_once(wrapper, frame):
    _cuda_sync()
    t0 = time.perf_counter()
    _ = wrapper.embed(frame)
    _cuda_sync()
    return (time.perf_counter() - t0) * 1000.0


# GLOBAL CACHE (same behavior as image version)
_cached_wrapper = None
_cached_model_name = None


def run_logic(model_name, iters, frame_h, frame_w, dataset, progress_callback=None):
    global _cached_wrapper, _cached_model_name

    # ---- Load the model only once ----
    if _cached_wrapper is None or _cached_model_name != model_name:
        _cached_wrapper = load_model(model_name)
        _cached_model_name = model_name
    wrapper = _cached_wrapper

    run_start = datetime.now().isoformat(timespec="seconds")

    # ---- no subject filtering (same as images) ----
    selected_env = os.getenv("YTF_SELECTED_SUBJECTS", "")
    selected_subjects = [s.strip() for s in selected_env.split(",") if s.strip()]

    if selected_subjects:
        send_log(f"Filtering to selected subjects: {', '.join(selected_subjects)}")

    # ---- Collect ALL frames ----
    image_paths = []
    for root, _, files in os.walk(dataset):
        for f in files:
            if f.lower().endswith(".jpg"):
                image_paths.append(os.path.join(root, f))

    image_paths.sort()

    if not image_paths:
        send_log("No .jpg frames found", "error")
        return

    total_images = len(image_paths)
    send_log(f"[CONFIG] Images={total_images}")

    # ---- override iters (like image version) ----
    if iters <= 0 or iters > total_images:
        iters = total_images
        send_log(f"iters=0 → using full dataset ({iters} frames)")

    # ---- Warmup ----
    first_frame = cv2.imread(image_paths[0])
    if first_frame is None:
        send_log("Could not read first frame", "error")
        return

    warm = 5 if (_HAS_TORCH and torch.cuda.is_available()) else 1
    send_log(f"🔥 Performing {warm} warm-up iteration(s)")

    for _ in range(warm):
        wrapper.embed(first_frame)
        _cuda_sync()

    # ---- Run Benchmark ----
    num_runs = int(os.getenv("YTF_RUNS", "1"))
    avg_runs = []
    all_runs = []

    for r in range(num_runs):
        send_log(f"--- Run {r+1}/{num_runs} ---")
        latencies = []

        for i, img_path in enumerate(image_paths[:iters], 1):
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            t_ms = measure_once(wrapper, frame)
            latencies.append(t_ms)

            # ---- Terminal progress bar + JSON ----
            if progress_callback:
                percent = int((i / iters) * 100)
                progress_callback(i, iters, r + 1, num_runs)
                print_progress_bar(percent, r + 1, num_runs)

        avg_ms = float(np.mean(latencies))
        avg_runs.append(avg_ms)
        all_runs.append(latencies)

        send_log(f"[Run {r+1}] Avg={avg_ms:.2f} ms, {1000/avg_ms:.2f} FPS")

    # ---- Final summary ----
    overall_avg = float(np.mean(avg_runs))
    run_end = datetime.now().isoformat(timespec="seconds")

    payload = {
        "source_file": "logic_dataset_video.py",
        "kind": "latency_video",
        "dataset": dataset,
        "num_frames": iters,
        "num_runs": num_runs,
        "avg_latency_ms": overall_avg,
        "avg_fps": 1000.0 / overall_avg,
        "model": model_name,
        "start_time": run_start,
        "end_time": run_end,
    }

    return payload
