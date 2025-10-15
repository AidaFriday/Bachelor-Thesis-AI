# ==== logic_dataset_video.py ====
import json, sys, time, numpy as np
import os, cv2
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

    # ✅ Warmup
    warmup = cv2.imread(image_paths[0])
    _ = wrapper.embed(warmup)
    _cuda_sync()

    # --- Multiple runs ---
    latency_series_all = []  # per-frame latency per run
    avg_latency_runs = []

    for r in range(num_runs):
        send_log(f"--- Run {r+1}/{num_runs} ---")
        latencies = []

        for i, img_path in enumerate(image_paths[:iters]):
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            t_ms = measure_once(wrapper, frame)
            latencies.append(t_ms)

            if (i + 1) % 10 == 0:
                send_log(f"Processed frame {i+1}/{iters} (run {r+1})")

        if not latencies:
            send_log(f"❌ No valid frames processed in run {r+1}", "error")
            continue

        avg_ms = float(np.mean(latencies))
        avg_latency_runs.append(avg_ms)
        latency_series_all.append(latencies)

        fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        send_log(f"[Run {r+1}] {iters} frames → {fps:.2f} FPS")

    # --- Final results ---
    if not avg_latency_runs:
        send_log("❌ All runs failed", "error")
        return

    avg_all_ms = float(np.mean(avg_latency_runs))
    avg_all_fps = 1000.0 / avg_all_ms if avg_all_ms > 0 else 0

    send_log
    (f"✅ Overall Avg Latency = {avg_all_ms:.2f} ms", "result")

    # ✅ Prepare GUI-compatible payload (latency not FPS)
    payload = {
        "kind": "latency_video",
        "dataset": dataset,
        "subjects": selected_subjects,
        "num_runs": num_runs,
        "iters": iters,
        "avg_latency_ms": avg_all_ms,
        "avg_fps": avg_all_fps,
        "latency_series_all": latency_series_all,  # ← per-frame latency
        "runs": avg_latency_runs,  # per-run average latency
        "model": model_name,
    }

    print(json.dumps(payload))
    sys.stdout.flush()
