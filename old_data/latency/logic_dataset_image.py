# ==== logic_dataset_image.py ====
import json, sys, time, numpy as np, os, cv2
from datetime import datetime
from connector import load_model

try:
    import torch
    _HAS_TORCH = True
except Exception:
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


# GLOBAL MODEL CACHE
_cached_wrapper = None
_cached_model_name = None


def run_logic(model_name, iters, frame_h, frame_w, dataset, progress_callback=None):
    global _cached_wrapper, _cached_model_name

    # Load model if needed
    if _cached_wrapper is None or _cached_model_name != model_name:
        _cached_wrapper = load_model(model_name)
        _cached_model_name = model_name
    wrapper = _cached_wrapper

    run_start = datetime.now().isoformat(timespec="seconds")

    start_person = os.getenv("LFW_START_PERSON", "")
    img_limit = int(os.getenv("LFW_IMAGE_COUNT", "0"))
    num_runs = int(os.getenv("LFW_RUNS", "1"))

    if not os.path.isdir(dataset):
        send_log(f"Invalid dataset path: {dataset}", "error")
        return

    # -------- Collect dataset --------
    people = sorted([d for d in os.listdir(dataset)
                     if os.path.isdir(os.path.join(dataset, d))])
    if not people:
        send_log("No folders found in dataset", "error")
        return

    # Start index
    try:
        start_idx = people.index(start_person)
    except ValueError:
        start_idx = 0

    # Gather image paths
    image_paths = []
    for person in people[start_idx:]:
        pdir = os.path.join(dataset, person)
        imgs = sorted([
            os.path.join(pdir, f)
            for f in os.listdir(pdir)
            if f.lower().endswith(".jpg")
        ])
        image_paths.extend(imgs)
        if img_limit and len(image_paths) >= img_limit:
            break

    if not image_paths:
        send_log("No images found", "error")
        return

    if img_limit and len(image_paths) > img_limit:
        image_paths = image_paths[:img_limit]

    total_images = len(image_paths)
    send_log(f"[CONFIG] Images={total_images}, Runs={num_runs}")

    # ---- Warmup ----
    first_frame = cv2.imread(image_paths[0])
    if first_frame is None:
        send_log("Cannot read first image", "error")
        return

    warm = 5 if (_HAS_TORCH and torch.cuda.is_available()) else 1
    send_log(f"🔥 Performing {warm} warm-up iteration(s)")

    for _ in range(warm):
        wrapper.embed(first_frame)
        _cuda_sync()

    # -------- Runs --------
    all_runs = []
    avg_runs = []
    frame_paths_all = []

    for r in range(num_runs):
        send_log(f"--- Run {r+1}/{num_runs} ---")
        latencies = []

        for i, img_path in enumerate(image_paths, 1):
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            t_ms = measure_once(wrapper, frame)
            latencies.append(t_ms)

            # ---- CLEAN progress reporting ----
            if progress_callback:
                percent = int((i / total_images) * 100)
                print_progress_bar(percent, r + 1, num_runs)


        if not latencies:
            send_log(f"Run {r+1} failed", "warn")
            continue

        avg_ms = float(np.mean(latencies))
        avg_runs.append(avg_ms)
        all_runs.append(latencies)
        frame_paths_all.append(image_paths.copy())

        send_log(f"[Run {r+1}] Avg={avg_ms:.2f} ms, {1000/avg_ms:.2f} FPS")

    if not all_runs:
        send_log("All runs failed", "error")
        return

    overall_avg = float(np.mean(avg_runs))
    run_end = datetime.now().isoformat(timespec="seconds")

    payload = {
        "kind": "latency_image",
        "dataset": dataset,
        "num_images": len(image_paths),
        "num_runs": num_runs,
        "avg_latency_ms": overall_avg,
        "model": model_name,
        "start_time": run_start,
        "end_time": run_end,
    }



    #print(json.dumps(payload), flush=True)
    return payload
