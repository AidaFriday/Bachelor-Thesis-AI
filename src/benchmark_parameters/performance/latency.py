import argparse, json, os, sys, time
import numpy as np
import cv2

# ---- Bootstrap sys.path so project root is importable ----
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from connector import load_model
from dataset import LFW  # ✅ use LFW loader

try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")


# ---------------- Helpers ----------------
def send_log(msg, level="info"):
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
    _ = wrapper.embed(frame)  # ✅ embeddings only
    _cuda_synchronize_if_needed()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0  # ms


def run(model_name, iters, frame_h, frame_w, dataset=None):
    wrapper = load_model(model_name)

    send_log(
        f"Running Latency benchmark | Model: {model_name} | "
        f"Dataset: {dataset or 'synthetic'} | Mode: embed_only"
    )

    frames = []
    image_map = {}

    # ✅ Load dataset images if available
    if dataset and dataset.lower().endswith("lfw") or "lfw" in (dataset or "").lower():
        images = LFW.list_all_images(
            root_dir=dataset, limit=iters, shuffle=True, verbose=False
        )
        for idx, path in enumerate(images, 1):
            img = cv2.imread(path)
            if img is not None:
                frames.append(img)
                image_map[idx] = os.path.basename(path)
                send_log(f"[{idx:03d}] Loaded dataset image: {path}")

    times = []
    for i in range(iters):
        if frames:  # dataset images
            frame = frames[i % len(frames)]
        else:  # synthetic
            frame = _random_frame(frame_h, frame_w)

        t = measure_once(wrapper, frame=frame)
        times.append(t)

        if (i + 1) % 5 == 0 or (i + 1) == iters:
            send_log(f"Processed {i+1}/{iters} images")

    stats = _percentiles(times)
    avg_ms = float(np.mean(times)) if times else float("nan")

    send_log(
        f"Latency summary (ms): "
        f"avg={avg_ms:.2f}, p50={stats['p50_ms']:.2f}, "
        f"p90={stats['p90_ms']:.2f}, p95={stats['p95_ms']:.2f}, "
        f"p99={stats['p99_ms']:.2f}, min={stats['min_ms']:.2f}, "
        f"max={stats['max_ms']:.2f}, std={stats['std_ms']:.2f}",
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

    if dataset and frames:
        mapping_file = os.path.join(PROJECT_ROOT, "latency_index_map.json")
        with open(mapping_file, "w") as f:
            json.dump(image_map, f, indent=4)
        send_log(f"Saved image index mapping to {mapping_file}")

    print(json.dumps(payload))
    sys.stdout.flush()


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
    parser.add_argument(
        "--iters", type=int, default=50, help="Number of iterations (default: 50)"
    )
    parser.add_argument(
        "--frame-size", type=str, default="640x640", help="HxW synthetic frames"
    )
    args = parser.parse_args()

    cfg = _resolve_settings()
    model = args.model or cfg.get("model")
    dataset = cfg.get("dataset")

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
