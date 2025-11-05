# ==== logic_dataset_image.py ====
import json, sys, time, numpy as np, os, cv2
from datetime import datetime
from connector import load_model

try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def send_log(msg, level="info"):
    print(json.dumps({"log": msg, "level": level}))
    sys.stdout.flush()


def _cuda_sync():
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_once(wrapper, frame):
    _cuda_sync()  # Ensures all GPU operations are finished before and after the measurement
    t0 = (
        time.perf_counter()
    )  # Records the precise high-resolution time (in seconds) before the embedding operation starts
    _ = wrapper.embed(
        frame
    )  # Runs the actual model inference or feature extraction on a single image/video frame
    _cuda_sync()  # Waits for all GPU work to finish — ensures we measure complete inference time
    return (
        time.perf_counter() - t0
    ) * 1000.0  # Subtracts start time from end time → elapsed seconds per frame


import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from connector import load_model

# global singleton
_cached_wrapper = None
_cached_model_name = None


def run_logic(model_name, iters, frame_h, frame_w, dataset):
    global _cached_wrapper, _cached_model_name
    if _cached_wrapper is None or _cached_model_name != model_name:
        _cached_wrapper = load_model(model_name)
        _cached_model_name = model_name
    wrapper = _cached_wrapper
    # mark overall run start time (ISO 8601)
    run_start = datetime.now().isoformat(timespec="seconds")

    start_person = os.getenv("LFW_START_PERSON", "")
    img_limit = int(os.getenv("LFW_IMAGE_COUNT", "0"))
    num_runs = int(os.getenv("LFW_RUNS", "1"))

    if not os.path.isdir(dataset):
        send_log(f"❌ Invalid dataset path: {dataset}", "error")
        return

    # ---- Step 1: collect all people in alphabetical order ----
    people = sorted(
        [d for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))]
    )
    if not people:
        send_log("❌ No person folders found", "error")
        return

    # ---- Step 2: find start index ----
    try:
        start_idx = people.index(start_person)
    except ValueError:
        start_idx = 0
        send_log(f"⚠️ Person '{start_person}' not found. Starting from '{people[0]}'")

    # ---- Step 3: gather image paths ----
    image_paths = []
    for person in people[start_idx:]:
        person_dir = os.path.join(dataset, person)
        imgs = sorted(
            [
                os.path.join(person_dir, f)
                for f in os.listdir(person_dir)
                if f.lower().endswith(".jpg")
            ]
        )
        image_paths.extend(imgs)
        if img_limit and len(image_paths) >= img_limit:
            break

    if not image_paths:
        send_log("❌ No images found in selected range", "error")
        return

    # clip to limit
    if img_limit and len(image_paths) > img_limit:
        image_paths = image_paths[:img_limit]

    send_log(
        f"[CONFIG] Start='{start_person}', Count={len(image_paths)}, Runs={num_runs}"
    )
    # ✅ Show image names only once
    send_log("🖼️ Loaded images:")
    for p in image_paths:
        send_log(f" - {os.path.basename(p)}")
    send_log(f"Total images: {len(image_paths)}")

    # ---- Adaptive Warmup ----
    first_frame = cv2.imread(image_paths[0])
    if first_frame is None:
        send_log("❌ Could not read first image for warm-up", "error")
        return

    # Decide warm-up count based on device
    if _HAS_TORCH and torch.cuda.is_available():
        warmup_iters = 5  # GPUs need more iterations to stabilize kernels
        device_name = "GPU"
    else:
        warmup_iters = 1  # CPU warm-up is mostly negligible
        device_name = "CPU"

    send_log(f"🔥 Performing {warmup_iters} warm-up iteration(s) on {device_name}")

    for _ in range(warmup_iters):
        _ = wrapper.embed(first_frame)
        _cuda_sync()

    # ---- Runs ----
    all_runs = []
    avg_runs = []
    frame_paths_all = []  # collect image paths per run
    for r in range(num_runs):
        send_log(f"--- Run {r+1}/{num_runs} ---")
        latencies = []

        for i, img_path in enumerate(image_paths):
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            t_ms = measure_once(wrapper, frame)
            latencies.append(t_ms)

            # ✅ Send live progress updates every 5 images or at the end
            if (i + 1) % 5 == 0 or (i + 1) == len(image_paths):
                progress_msg = {
                    "_type": "progress",
                    "progress": i + 1,
                    "total": len(image_paths),
                    "run": r + 1,  # include current run number
                    "num_runs": num_runs,  # include total number of runs
                }
                print(json.dumps(progress_msg), flush=True)

        if not latencies:
            send_log(f"⚠️ Run {r+1} had no valid images", "warn")
            continue

        avg_ms = float(np.mean(latencies))
        avg_runs.append(avg_ms)
        all_runs.append(latencies)
        frame_paths_all.append(image_paths.copy())  # keep file list for this run

        send_log(f"[Run {r+1}] Avg={avg_ms:.2f} ms, {1000/avg_ms:.2f} FPS")

    if not all_runs:
        send_log("❌ All runs failed", "error")
        return

    overall_avg = float(np.mean(avg_runs))
    # mark end time
    run_end = datetime.now().isoformat(timespec="seconds")

    # --- Build JSON payload ---
    payload = {
        "source_file": os.path.basename(__file__),  # ✅ Add the file name
        "kind": "latency_image",
        "dataset": dataset,
        "start_person": start_person,
        "start_identity": start_person,
        "num_images": len(image_paths),
        "num_runs": num_runs,
        "avg_latency_ms": overall_avg,
        "latency_series_all": all_runs,
        "image_paths": image_paths,
        "frame_paths_all": frame_paths_all,
        "model": model_name,
        "start_time": run_start,
        "end_time": run_end,
    }

    print(json.dumps(payload))
    sys.stdout.flush()
