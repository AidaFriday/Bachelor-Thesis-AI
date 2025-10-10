# ==== latency.py ====
import argparse
import json
import os
import sys
import time
import numpy as np

# ---------------------------------------------------------------------
# 🔹 Load project-wide index to dynamically locate all Python modules
# ---------------------------------------------------------------------
try:
    from components.utilities.file_indexer import load_file_index
except ModuleNotFoundError:
    # fallback for standalone run
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "../../../components/utilities")
    )
    from file_indexer import load_file_index

index_data = load_file_index()
PROJECT_ROOT = index_data["root"]

# Dynamically add every directory containing a Python file to sys.path
for rel_path in index_data["files"]:
    dir_path = os.path.join(PROJECT_ROOT, os.path.dirname(rel_path))
    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)

# ---------------------------------------------------------------------
# 🔹 Import project modules
# ---------------------------------------------------------------------
from connector import load_model  # ✅ now resolvable from index

try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")


# ---------------------------------------------------------------------
# 🔹 Helper utilities
# ---------------------------------------------------------------------
def send_log(msg, level="info"):
    """Send log message to GUI via JSON line."""
    payload = {"log": msg, "level": level}
    print(json.dumps(payload))
    sys.stdout.flush()


def _cuda_synchronize_if_needed():
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentiles(ms_list):
    arr = np.array(ms_list, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(arr, 50)) if arr.size else float("nan"),
        "p90_ms": float(np.percentile(arr, 90)) if arr.size else float("nan"),
        "p95_ms": float(np.percentile(arr, 95)) if arr.size else float("nan"),
        "p99_ms": float(np.percentile(arr, 99)) if arr.size else float("nan"),
        "min_ms": float(np.min(arr)) if arr.size else float("nan"),
        "max_ms": float(np.max(arr)) if arr.size else float("nan"),
        "std_ms": float(np.std(arr, ddof=0)) if arr.size else float("nan"),
    }


def _random_frame(h=640, w=640):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def measure_once(wrapper, frame=None):
    """Single latency measurement (embedding only)."""
    _cuda_synchronize_if_needed()
    t0 = time.perf_counter()
    _ = wrapper.embed(frame)
    _cuda_synchronize_if_needed()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0  # ms


def measure_embed_only(wrapper, iters=1, frame_h=640, frame_w=640):
    """
    Used by inference.py to benchmark model embedding latency.
    Returns a list of per-iteration latencies in milliseconds.
    """
    frame = _random_frame(frame_h, frame_w)
    _ = wrapper.embed(frame)  # warm-up
    _cuda_synchronize_if_needed()

    times_ms = []
    for _ in range(iters):
        times_ms.append(measure_once(wrapper, frame=frame))
    return times_ms


# ---------------------------------------------------------------------
# 🔹 Main benchmark logic
# ---------------------------------------------------------------------
def run(model_name, iters, frame_h, frame_w, dataset=None):
    wrapper = load_model(model_name)

    send_log(
        f"Running Latency benchmark | Model: {model_name} | "
        f"Dataset: {dataset or 'synthetic'} | Mode: embed_only"
    )

    # Warm-up
    warmup_frame = _random_frame(frame_h, frame_w)
    send_log("Running warm-up inferences (not timed)...")
    for _ in range(10):
        _ = wrapper.embed(warmup_frame)
        _cuda_synchronize_if_needed()

    # Measurement loop
    times = []
    for i in range(iters):
        t = measure_once(wrapper, frame=_random_frame(frame_h, frame_w))
        times.append(t)
        if (i + 1) % 5 == 0 or (i + 1) == iters:
            send_log(f"Processed {i+1}/{iters} frames")

    stats = _percentiles(times)
    avg_ms = float(np.mean(times)) if times else float("nan")

    send_log(
        f"Latency summary (ms): avg={avg_ms:.2f}, "
        f"p50={stats['p50_ms']:.2f}, p90={stats['p90_ms']:.2f}, "
        f"p95={stats['p95_ms']:.2f}, p99={stats['p99_ms']:.2f}, "
        f"min={stats['min_ms']:.2f}, max={stats['max_ms']:.2f}, "
        f"std={stats['std_ms']:.2f}",
        level="result",
    )

    payload = {
        "kind": "latency",
        "model": model_name,
        "mode": "embed_only",
        "dataset": dataset or "synthetic",
        "avg_ms": avg_ms,
        "times": times,
        **stats,
    }

    print(json.dumps(payload))
    sys.stdout.flush()


# ---------------------------------------------------------------------
# 🔹 Settings & CLI entry point
# ---------------------------------------------------------------------
def _resolve_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="arcface | facenet | insightface")
    parser.add_argument("--iters", type=int, default=50, help="Number of iterations")
    parser.add_argument("--frame-size", type=str, default="640x640", help="HxW size")
    parser.add_argument(
        "--dataset", type=str, default=None, help="Optional dataset path"
    )
    args = parser.parse_args()

    cfg = _resolve_settings()
    model = args.model or cfg.get("model")
    dataset = args.dataset or cfg.get("dataset")

    if not model:
        print(json.dumps({"error": "No model selected"}))
        sys.exit(1)

    try:
        h, w = [int(p) for p in args.frame_size.lower().split("x")]
    except Exception:
        h, w = 640, 640

    run(model, args.iters, h, w, dataset=dataset)


if __name__ == "__main__":
    main()
