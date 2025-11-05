# ==== inference.py ====

import argparse
import json
import os
import sys
import numpy as np

# ---------------------------------------------------------------------
# 🔹 Load centralized project file index (instead of manual sys.path setup)
# ---------------------------------------------------------------------


# add near the top of logic_inference.py
try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _cuda_sync():
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


def _get_embed_wh(wrapper):
    # Prefer explicit (W,H) if wrapper exposes it
    wh = getattr(wrapper, "_embed_wh", None)
    if wh and len(wh) == 2:
        return int(wh[0]), int(wh[1])
    # Otherwise try _rec_input (varies by libs) and normalize to (W,H)
    rec = getattr(wrapper, "_rec_input", None)
    if rec and len(rec) == 2:
        H, W = int(rec[0]), int(rec[1])  # many libs expose (H, W)
        return (W, H)
    # Finally fall back to input_size; treat as (H,W) by convention
    sz = getattr(wrapper, "input_size", (112, 112))
    H, W = int(sz[0]), int(sz[1])
    return (W, H)


def measure_embed_only(wrapper, iters=1, size=None):
    import time, numpy as np, cv2

    if size is None:
        W, H = _get_embed_wh(wrapper)  # always normalized to (W,H)
    else:
        W, H = int(size[0]), int(size[1])

    times = []
    for _ in range(iters):
        dummy = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        _cuda_sync()
        t0 = time.perf_counter()
        _ = wrapper.embed(dummy)
        _cuda_sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


try:
    from components.utilities.file_indexer import load_file_index
except ModuleNotFoundError:
    # fallback if running standalone (e.g., without full project context)
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "../../../components/utilities")
    )
    from file_indexer import load_file_index

# Load the generated file index
index_data = load_file_index()
PROJECT_ROOT = index_data["root"]

# Automatically add all directories with Python files to sys.path
for rel_path in index_data["files"]:
    dir_path = os.path.join(PROJECT_ROOT, os.path.dirname(rel_path))
    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)

# ---------------------------------------------------------------------
# 🔹 Import project modules
# ---------------------------------------------------------------------
from connector import load_model

SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")


# ---------------------------------------------------------------------
# 🔹 Helper Functions
# ---------------------------------------------------------------------
def send_log(msg, level="info"):
    """Send structured log messages to stdout (for GUI or logging system)."""
    payload = {"log": msg, "level": level}
    print(json.dumps(payload))
    sys.stdout.flush()


def _resolve_settings():
    """Read settings.json if available."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _summarize(times_ms):
    """Compute and return basic latency statistics."""
    if not times_ms:
        return dict(
            avg_ms=float("nan"),
            min_ms=float("nan"),
            max_ms=float("nan"),
            p50_ms=float("nan"),
            p90_ms=float("nan"),
            p95_ms=float("nan"),
            p99_ms=float("nan"),
            std_ms=float("nan"),
        )
    arr = np.array(times_ms, dtype=np.float64)
    return dict(
        avg_ms=float(np.mean(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p90_ms=float(np.percentile(arr, 90)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        std_ms=float(np.std(arr, ddof=0)),
    )


# ---------------------------------------------------------------------
# 🔹 Core Benchmark Logic
# ---------------------------------------------------------------------
def run(model_name: str, iters: int, dataset: str = None):
    """Perform `iters` inferences and log latency results in milliseconds."""
    wrapper = load_model(model_name)

    send_log(
        f"Running Inference benchmark | Model: {model_name} | "
        f"Dataset: {dataset or 'synthetic'} | Mode: embed_only"
    )

    times_ms = []
    for i in range(iters):
        t = measure_embed_only(wrapper, iters=1)
        times_ms.extend(t)

        if (i + 1) % 5 == 0 or (i + 1) == iters:
            send_log(f"Progress: {i + 1}/{iters} inferences completed")

    stats = _summarize(times_ms)

    payload = {
        "kind": "inference",
        "model": model_name,
        "mode": "embed_only",
        "dataset": dataset or "synthetic",
        "count": len(times_ms),
        "times": times_ms,
        **stats,
    }

    # Emit results to stdout as JSON for GUI or CLI consumption
    print(json.dumps(payload))
    sys.stdout.flush()


# ---------------------------------------------------------------------
# 🔹 Command-Line Interface Entry Point
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run model inference latency test.")
    parser.add_argument("--model", type=str, help="arcface | facenet | insightface")
    parser.add_argument(
        "--iters", type=int, default=50, help="Number of inferences to execute"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Optional dataset path"
    )

    args = parser.parse_args()

    cfg = _resolve_settings()
    model = args.model or cfg.get("model")
    dataset = args.dataset or cfg.get("dataset")

    if not model:
        print(
            json.dumps(
                {
                    "error": "No model selected. Use --model or define one in settings.json"
                }
            )
        )
        sys.exit(1)

    run(model, args.iters, dataset=dataset)


if __name__ == "__main__":
    main()
