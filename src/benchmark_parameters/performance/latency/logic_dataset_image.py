# ==== logic_dataset_image.py ====
import json, sys, time, numpy as np
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


def _random_frame(h=640, w=640):
    import numpy as np

    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def measure_once(wrapper, frame):
    _cuda_sync()
    t0 = time.perf_counter()
    _ = wrapper.embed(frame)
    _cuda_sync()
    return (time.perf_counter() - t0) * 1000.0


def run_logic(model_name, iters, frame_h, frame_w, dataset):
    wrapper = load_model(model_name)
    send_log(f"Running image latency test with {iters} iterations")

    frame = _random_frame(frame_h, frame_w)
    _ = wrapper.embed(frame)
    _cuda_sync()

    times = []
    for i in range(iters):
        frame = _random_frame(frame_h, frame_w)
        t = measure_once(wrapper, frame)
        times.append(t)
        if (i + 1) % 5 == 0:
            send_log(f"Processed {i+1}/{iters} frames")

    avg_ms = float(np.mean(times))
    stats = {
        "p50_ms": float(np.percentile(times, 50)),
        "p90_ms": float(np.percentile(times, 90)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "std_ms": float(np.std(times)),
    }

    send_log(f"Image Latency avg={avg_ms:.2f} ms", "result")

    payload = {
        "kind": "latency_image",
        "dataset": dataset,
        "avg_ms": avg_ms,
        "times": times,
        **stats,
    }
    print(json.dumps(payload))
    sys.stdout.flush()
