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


def _simulate_video_frame(h=640, w=640, frame_idx=0):
    """Simulate slight motion variation per frame."""
    frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    frame = np.roll(frame, shift=frame_idx % 5, axis=1)  # mimic small movement
    return frame


def measure_once(wrapper, frame):
    _cuda_sync()
    t0 = time.perf_counter()
    _ = wrapper.embed(frame)
    _cuda_sync()
    return (time.perf_counter() - t0) * 1000.0


def run_logic(model_name, iters, frame_h, frame_w, dataset):
    wrapper = load_model(model_name)

    # ✅ read selected subjects from environment (if set)
    selected_env = os.getenv("YTF_SELECTED_SUBJECTS", "")
    selected_subjects = [s.strip() for s in selected_env.split(",") if s.strip()]
    if selected_subjects:
        send_log(f"Filtering to selected subjects: {', '.join(selected_subjects)}")

    # ✅ load frames only for selected subjects
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

    # ✅ auto-adjust iteration count
    if iters <= 0 or iters > len(image_paths):
        iters = len(image_paths)
        send_log(f"⚠️ iters=0 detected → using {iters} frames from selected subjects")

    # ✅ number of test repetitions
    num_runs = int(os.getenv("YTF_RUNS", "1"))
    send_log(f"[CONFIG] Performing {num_runs} run(s) × {iters} frames each")

    # ✅ pre-load warmup
    warmup = cv2.imread(image_paths[0])
    _ = wrapper.embed(warmup)
    _cuda_sync()

    run_times = []  # average per run

    # --- Multiple runs ---
    for r in range(num_runs):
        send_log(f"--- Run {r+1}/{num_runs} ---")
        times = []

        for i, img_path in enumerate(image_paths[:iters]):
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            t = measure_once(wrapper, frame)
            times.append(t)

            if (i + 1) % 10 == 0:
                send_log(f"Processed frame {i+1}/{iters} (run {r+1})")

        if not times:
            send_log(f"❌ No valid frames processed in run {r+1}", "error")
            continue

        avg_ms = float(np.mean(times))
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        run_times.append(avg_ms)

        send_log(f"[Run {r+1}] {iters} frames → {fps:.2f} FPS")

    # --- Final results ---
    if not run_times:
        send_log("❌ All runs failed", "error")
        return

    avg_all_ms = float(np.mean(run_times))
    avg_all_fps = 1000.0 / avg_all_ms if avg_all_ms > 0 else 0

    send_log(
        f"✅ Overall Avg Latency = {avg_all_ms:.2f} ms → {avg_all_fps:.2f} FPS",
        "result",
    )

    payload = {
        "kind": "latency_video",
        "dataset": dataset,
        "subjects": selected_subjects,
        "num_runs": num_runs,
        "iters": iters,
        "avg_latency_ms": avg_all_ms,
        "avg_fps": avg_all_fps,
        "run_times_ms": run_times,
    }
    print(json.dumps(payload))
    sys.stdout.flush()
