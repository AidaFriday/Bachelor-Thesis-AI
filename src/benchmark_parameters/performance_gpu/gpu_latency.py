# ==== latency.py ====

import argparse
import json
import os
import sys
import time
import numpy as np

# -------------------------------------------------------------
# 🔥 FORCE PROJECT ROOT INTO PYTHONPATH
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
COMPONENTS_DIR = os.path.join(PROJECT_ROOT, "components")

for p in [PROJECT_ROOT, COMPONENTS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# -------------------------------------------------------------
# 🔹 GPU CHECK
# -------------------------------------------------------------
def _gpu_available():
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass

    try:
        import tensorflow as tf
        if len(tf.config.list_physical_devices("GPU")) > 0:
            return True
    except Exception:
        pass

    if any(os.environ.get(v) for v in ["CUDA_PATH", "CUDA_HOME", "NVIDIA_VISIBLE_DEVICES"]):
        return True

    return False

if not _gpu_available():
    print(json.dumps({"error": "No GPU available — script cancelled"}), flush=True)
    sys.exit(1)

# -------------------------------------------------------------
# 🔹 LOAD PROJECT FILE INDEX
# -------------------------------------------------------------
try:
    from components.utilities.file_indexer import load_file_index
except ModuleNotFoundError:
    try:
        from file_indexer import load_file_index
    except Exception as e:
        print(json.dumps({"error": f"file_indexer import failed: {e}"}), flush=True)
        sys.exit(1)

index_data = load_file_index()
PROJECT_ROOT = index_data["root"]

for rel in index_data["files"]:
    d = os.path.join(PROJECT_ROOT, os.path.dirname(rel))
    if d not in sys.path:
        sys.path.insert(0, d)

from connector import load_model

SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")

# -------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------
def send_log(msg, level="info"):
    print(json.dumps({"log": msg, "level": level}), flush=True)

# -------------------------------------------------------------
# CLEAN PROGRESS EVENT
# -------------------------------------------------------------
_last_progress = -1

# -------------------------------------------------------------
# CLEAN PROGRESS EVENT + ETA
# -------------------------------------------------------------
_last_progress = -1
_start_time_progress = None

def send_progress(current, total, run=1, num_runs=1):
    global _last_progress, _start_time_progress

    if _start_time_progress is None:
        _start_time_progress = time.time()

    pct = int((current / total) * 100)

    if pct != _last_progress:
        _last_progress = pct

        elapsed = time.time() - _start_time_progress
        if current > 0:
            rate = elapsed / current
            remaining = rate * (total - current)
        else:
            remaining = None

        payload = {
            "_type": "progress",
            "progress": current,
            "total": total,
            "percent": pct,
            "run": run,
            "num_runs": num_runs,
            "elapsed_sec": round(elapsed, 2),
            "eta_sec": round(remaining, 2) if remaining is not None else None
        }

        print(json.dumps(payload), flush=True)

# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
def _resolve_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            return json.load(open(SETTINGS_FILE, "r"))
        except:
            pass
    return {}

# -------------------------------------------------------------
# DATASET NORMALIZATION
# -------------------------------------------------------------
def normalize_dataset_path(dataset):
    if not dataset:
        return dataset

    dataset = os.path.abspath(os.path.expanduser(dataset))

    if not os.path.exists(dataset):
        send_log(f"Invalid dataset path: {dataset}", "error")
        sys.exit(1)

    # Auto select LFW deepfunneled
    if "lfw" in dataset.lower():
        deep = os.path.join(dataset, "lfw-deepfunneled")
        if os.path.exists(deep):
            dataset = deep

    return dataset

# -------------------------------------------------------------
# ROUTER
# -------------------------------------------------------------
def run(model_name, iters, frame_h, frame_w, dataset):

    dataset = normalize_dataset_path(dataset)
    ds = (dataset or "").lower()

    if "lfw" in ds:
        from benchmark_parameters.performance.latency.logic_dataset_image import run_logic
        send_log(f"[latency] Using IMAGE logic → {dataset}")

    elif "ytf" in ds or "aligned" in ds or "video" in ds:
        from benchmark_parameters.performance.latency.logic_dataset_video import run_logic
        send_log(f"[latency] Using VIDEO logic → {dataset}")

    else:
        from benchmark_parameters.performance.latency.logic_dataset_image import run_logic
        send_log(f"[latency] Defaulting to IMAGE logic")

    # Important: run_logic must return result payload
    return run_logic(model_name, iters, frame_h, frame_w, dataset, send_progress)

# -------------------------------------------------------------
# MAIN ENTRY
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="arcface | facenet | insightface")
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--frame-size", type=str, default="640x640")
    parser.add_argument("--dataset", type=str)

    args = parser.parse_args()
    cfg = _resolve_settings()

    model = args.model or cfg.get("model")
    dataset = args.dataset or cfg.get("dataset")

    if not model:
        print(json.dumps({"error": "No model selected"}), flush=True)
        sys.exit(1)

    try:
        h, w = [int(x) for x in args.frame_size.lower().split("x")]
    except:
        h, w = 640, 640

    start = time.time()

    iters = args.iters if args.iters > 0 else 0
    result_payload = run(model, iters, h, w, dataset)

    end = time.time()

    # -------------------------------------------------------------
    # CALCULATE LATENCY STATISTICS
    # -------------------------------------------------------------
    lat_list = []

    if isinstance(result_payload, dict):
        for key in ["latencies", "latency", "times"]:
            if key in result_payload and isinstance(result_payload[key], (list, tuple)):
                lat_list = result_payload[key]
                break

    if lat_list:
        avg_latency = float(np.mean(lat_list))
        min_latency = float(np.min(lat_list))
        max_latency = float(np.max(lat_list))
    else:
        avg_latency = None
        min_latency = None
        max_latency = None

    result_payload["avg_latency"] = avg_latency
    result_payload["min_latency"] = min_latency
    result_payload["max_latency"] = max_latency

    # -------------------------------------------------------------
    # FINAL RESULT OBJECT
    # -------------------------------------------------------------
    final = {
        "model": model,
        "dataset": dataset,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start)),
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(end)),
        "duration_sec": round(end - start, 2),
        "result": result_payload,
    }

    # -------------------------------------------------------------
    # SAVE RESULT TO FILE (DELETE OLD → CREATE NEW)
    # -------------------------------------------------------------
    output_file = os.path.join(CURRENT_DIR, "latency_result.txt")

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, "w") as f:
        f.write(json.dumps(final, indent=2))

    send_log(f"Saved final results to {output_file}")

    # Also print final JSON to stdout
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
