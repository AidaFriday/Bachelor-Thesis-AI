# latency.py
import argparse, json, os, sys, time
import numpy as np

# ---- Bootstrap sys.path so project root is importable ----
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from connector import load_model

# Optional: only imported if available (for CUDA sync later on GPU)
try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")


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
        "std_ms": float(np.std(arr)) if arr.size else float("nan"),
    }


def _random_frame(h=640, w=640):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def measure_detect_and_embed(wrapper, iters=50, frame_h=640, frame_w=640):
    # Warm-up (don’t record)
    for _ in range(5):
        _ = wrapper.detect_and_embed(_random_frame(frame_h, frame_w))

    times_ms = []
    for _ in range(iters):
        frame = _random_frame(frame_h, frame_w)
        _cuda_synchronize_if_needed()
        t0 = time.perf_counter()
        _ = wrapper.detect_and_embed(frame)
        _cuda_synchronize_if_needed()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    return times_ms


def measure_embed_only(wrapper, iters=50):
    """
    Pure model forward where possible.
    NOTE: in your wrappers, FaceNet's embed() is a true forward pass.
          ArcFace/InsightFace embed() still triggers detection.
    """
    h, w = getattr(wrapper, "input_size", (160, 160))
    for _ in range(5):
        _ = wrapper.embed(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))

    times_ms = []
    for _ in range(iters):
        x = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        _cuda_synchronize_if_needed()
        t0 = time.perf_counter()
        _ = wrapper.embed(x)
        _cuda_synchronize_if_needed()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    return times_ms


def run(model_name, iters, target, frame_h, frame_w):
    wrapper = load_model(model_name)

    if target == "detect":
        times = measure_detect_and_embed(
            wrapper, iters=iters, frame_h=frame_h, frame_w=frame_w
        )
        stats = _percentiles(times)
        payload = {
            "model": model_name,
            "mode": "detect_and_embed",
            "avg_ms": float(np.mean(times)) if times else float("nan"),
            "times": times,
            **stats,
        }
        print(json.dumps(payload))
        return

    if target == "embed":
        times = measure_embed_only(wrapper, iters=iters)
        stats = _percentiles(times)
        payload = {
            "model": model_name,
            "mode": "embed_only",
            "avg_ms": float(np.mean(times)) if times else float("nan"),
            "times": times,
            **stats,
        }
        print(json.dumps(payload))
        return

    # both → keep detect as the primary series, include embed stats in details
    det_times = measure_detect_and_embed(
        wrapper, iters=iters, frame_h=frame_h, frame_w=frame_w
    )
    emb_times = measure_embed_only(wrapper, iters=iters)
    det_stats = _percentiles(det_times)
    emb_stats = _percentiles(emb_times)
    payload = {
        "model": model_name,
        "mode": "both",
        "times": det_times,  # primary for plotting
        "avg_ms": float(np.mean(det_times)) if det_times else float("nan"),
        **det_stats,
        "details": {
            "detect_and_embed": {
                "avg_ms": float(np.mean(det_times)) if det_times else float("nan"),
                "times": det_times,
                **det_stats,
            },
            "embed_only": {
                "avg_ms": float(np.mean(emb_times)) if emb_times else float("nan"),
                "times": emb_times,
                **emb_stats,
            },
        },
    }
    print(json.dumps(payload))


def _resolve_model_from_settings(default=None):
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                cfg = json.load(f)
            return cfg.get("model", default)
        except Exception:
            return default
    return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="arcface | facenet | insightface")
    parser.add_argument(
        "--iters", type=int, default=50, help="Number of iterations (default: 50)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="detect",
        choices=["detect", "embed", "both"],
        help="detect=end-to-end, embed=model-only, both=report both",
    )
    parser.add_argument(
        "--frame-size",
        type=str,
        default="640x640",
        help="HxW for detect mode synthetic frames (default: 640x640)",
    )
    args = parser.parse_args()

    model = args.model or _resolve_model_from_settings()
    if not model:
        print(
            json.dumps(
                {"error": "No model selected. Pass --model or set it in settings.json"}
            )
        )
        sys.exit(1)

    try:
        h, w = [int(p) for p in args.frame_size.lower().split("x")]
    except Exception:
        h, w = 640, 640

    run(model, args.iters, args.target, h, w)


if __name__ == "__main__":
    main()
