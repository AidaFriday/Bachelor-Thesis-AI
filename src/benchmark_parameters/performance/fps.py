# fps.py
import argparse, json, os, sys
import numpy as np

# ---- Bootstrap sys.path so project root is importable ----
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from connector import load_model

# Reuse timing helpers from latency.py (warmup + optional CUDA sync handled there)
from latency import measure_detect_and_embed, measure_embed_only

SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")


def _resolve_model_from_settings(default=None):
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                cfg = json.load(f)
            return cfg.get("model", default)
        except Exception:
            return default
    return default


def run(model_name: str, iters: int, target: str, frame_h: int, frame_w: int):
    """
    target: 'detect' -> end-to-end detect+embed on synthetic HxW frames
            'embed'  -> model-only (true forward only for FaceNet)
    """
    wrapper = load_model(model_name)

    if target == "detect":
        times_ms = measure_detect_and_embed(
            wrapper, iters=iters, frame_h=frame_h, frame_w=frame_w
        )
        mode = "detect_and_embed"
    else:  # 'embed'
        times_ms = measure_embed_only(wrapper, iters=iters)
        mode = "embed_only"

    # Convert each latency sample to FPS and summarize
    fps_series = [1000.0 / t if t > 0 else float("inf") for t in times_ms]
    mean_fps = float(np.mean(fps_series)) if fps_series else float("nan")

    # Output FPS-only payload (no latency keys)
    payload = {
        "model": model_name,
        "mode": mode,
        "fps": mean_fps,
        "fps_series": fps_series,
    }
    print(json.dumps(payload))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, help="arcface | facenet | insightface")
    ap.add_argument(
        "--iters", type=int, default=50, help="Number of iterations (default: 50)"
    )
    ap.add_argument(
        "--target",
        choices=["detect", "embed"],
        default="detect",
        help="detect=end-to-end pipeline, embed=model-only (pure for FaceNet)",
    )
    ap.add_argument(
        "--frame-size",
        default="640x640",
        help="HxW for detect mode synthetic frames (default: 640x640)",
    )
    args = ap.parse_args()

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
