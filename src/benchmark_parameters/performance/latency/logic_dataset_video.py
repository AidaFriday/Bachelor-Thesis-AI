# ==== logic_dataset_video.py ====
import json, sys, time, numpy as np
import os, cv2
from datetime import datetime  # ✅ NEW
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
    """Measure latency (ms) for a single frame."""
    _cuda_sync()
    t0 = time.perf_counter()
    _ = wrapper.embed(frame)
    _cuda_sync()
    return (time.perf_counter() - t0) * 1000.0  # milliseconds


def run_logic(model_name, iters, frame_h, frame_w, dataset):
    wrapper = load_model(model_name)

    # ✅ Selected subjects (from GUI dialog)
    selected_env = os.getenv("YTF_SELECTED_SUBJECTS", "")
    selected_subjects = [s.strip() for s in selected_env.split(",") if s.strip()]
    if selected_subjects:
        send_log(f"Filtering to selected subjects: {', '.join(selected_subjects)}")

    # ✅ NEW: remember the "starting identity" (first subject chosen in the GUI)
    start_identity = selected_subjects[0] if selected_subjects else None  # ✅ NEW

    # ✅ Gather all frames for these subjects
    image_paths = []
    for root, _, files in os.walk(dataset):
        if selected_subjects and not any(
            subj.lower() in root.lower() for subj in selected_subjects
        ):
            continue
        for f in files:
            if f.lower().endswith(".jpg"):
                image_paths.append(os.path.join(root, f))
    image_paths.sort()

    if not image_paths:
        send_log("❌ No .jpg frames found for selected subjects", "error")
        return

    # ✅ Determine iteration count
    if iters <= 0 or iters > len(image_paths):
        iters = len(image_paths)
        send_log(f"⚠️ iters=0 detected → using {iters} frames from selected subjects")

    # ✅ Number of test repetitions
    num_runs = int(os.getenv("YTF_RUNS", "1"))
    send_log(f"[CONFIG] Performing {num_runs} run(s) × {iters} frames each")

    # ✅ NEW: overall benchmark start time
    run_start = datetime.now().isoformat(timespec="seconds")  # ✅ NEW

    # ---- Adaptive Warmup ----
    first_frame = cv2.imread(image_paths[0]) if len(image_paths) > 0 else None
    if first_frame is None:
        send_log("❌ Could not read first frame for warm-up", "error")
        # fallback: create random frame if dataset read fails
        first_frame = np.random.randint(0, 255, (frame_h, frame_w, 3), dtype=np.uint8)

    # Decide warm-up count based on device
    if _HAS_TORCH and torch.cuda.is_available():
        warmup_iters = 5  # GPU needs multiple iterations to stabilize kernels
        device_name = "GPU"
    else:
        warmup_iters = 1  # CPU only needs one iteration
        device_name = "CPU"

    send_log(f"🔥 Performing {warmup_iters} warm-up iteration(s) on {device_name}")

    for _ in range(warmup_iters):
        _ = wrapper.embed(first_frame)
        _cuda_sync()

    # --- Multiple runs ---
    latency_series_all = []  # per-frame latency per run
    avg_latency_runs = []
    frame_paths_all = []  #  create inside function scope

    for r in range(num_runs):
        send_log(f"--- Run {r+1}/{num_runs} ---")
        latencies = []
        run_paths = []

        for i, img_path in enumerate(image_paths[:iters]):
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            t_ms = measure_once(wrapper, frame)
            latencies.append(t_ms)
            run_paths.append(img_path)

            # ✅ Send live progress every 10 frames or at the end
            if (i + 1) % 10 == 0 or (i + 1) == iters:
                progress_msg = {
                    "_type": "progress",
                    "progress": i + 1,
                    "total": iters,
                    "run": r + 1,
                    "num_runs": num_runs,
                }
                print(json.dumps(progress_msg), flush=True)

        if not latencies:
            send_log(f"❌ No valid frames processed in run {r+1}", "error")
            continue

        avg_ms = float(np.mean(latencies))
        avg_latency_runs.append(avg_ms)
        latency_series_all.append(latencies)
        frame_paths_all.append(run_paths)  # ✅ append paths for this run

        fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        send_log(f"[Run {r+1}] {iters} frames → {avg_ms:.2f} ms → {fps:.2f} FPS")

    # --- Final results ---
    if not avg_latency_runs:
        send_log("❌ All runs failed", "error")
        return

    avg_all_ms = float(np.mean(avg_latency_runs))
    avg_all_fps = 1000.0 / avg_all_ms if avg_all_ms > 0 else 0

    send_log(f"✅ Overall Avg Latency = {avg_all_ms:.2f} ms", "result")

    # ✅ NEW: overall benchmark end time
    run_end = datetime.now().isoformat(timespec="seconds")  # ✅ NEW

    # ✅ JSON payload: now includes model, starting identity, start & end time
    payload = {
        "source_file": os.path.basename(__file__),
        "kind": "latency_video",  # optional, just to distinguish
        "dataset": dataset,
        "subjects": selected_subjects,
        "start_identity": start_identity,
        "start_person": start_identity,
        "start_time": run_start,
        "end_time": run_end,  # ✅ add this line
        "num_runs": num_runs,
        "iters": iters,
        "avg_latency_ms": avg_all_ms,
        "avg_fps": avg_all_fps,
        "latency_series_all": latency_series_all,
        "runs": avg_latency_runs,
        "model": model_name,
        "frame_paths_all": frame_paths_all,
    }

    print(json.dumps(payload))
    sys.stdout.flush()
    return payload      # <--- ADD THIS
    