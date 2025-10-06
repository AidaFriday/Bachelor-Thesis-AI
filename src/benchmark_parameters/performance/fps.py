import argparse, json, os, sys
import numpy as np
import cv2
import time

# ---- Bootstrap sys.path so project root is importable ----
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from connector import load_model
from latency import measure_detect_and_embed, measure_embed_only

# Dataset loaders
from dataset import LFW  # add your own iphone dataset loader similarly

SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")


def _resolve_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def send_log(msg, level="info"):
    """Send a log message to GUI as JSON (progress logs)"""
    payload = {"log": msg, "level": level}
    print(json.dumps(payload))
    sys.stdout.flush()


def run(
    model_name: str,
    iters: int,
    target: str,
    frame_h: int,
    frame_w: int,
    dataset: str = None,
):
    """
    target: 'detect' -> detect+embed either on synthetic frames or dataset
            'embed'  -> embedding only
    dataset: optional dataset key (e.g., 'lfw', 'iphone16'), else synthetic
    """
    wrapper = load_model(model_name)

    send_log(
        f"Running FPS benchmark | Model: {model_name} | "
        f"Dataset: {dataset or 'synthetic'} | Mode: {target}"
    )

    frames = []
    if dataset and dataset.lower() == "lfw":
        pairs = LFW.load_pairs(limit=iters)
        frames = [cv2.imread(p[0]) for p in pairs[:iters] if os.path.exists(p[0])]
    elif dataset and dataset.lower() == "iphone16":
        # TODO: implement dataset/iphone16.py loader
        pass

    times_ms = []
    start = time.time()

    if target == "detect":
        if frames:
            for i, img in enumerate(frames, 1):
                if img is None:
                    continue
                h, w = frame_h, frame_w
                resized = cv2.resize(img, (w, h))
                t = measure_detect_and_embed(
                    wrapper, iters=1, frame_h=h, frame_w=w, override_frame=resized
                )
                times_ms.extend(t)

                if i % 5 == 0 or i == len(frames):
                    send_log(f"Processed {i}/{len(frames)} frames…")
        else:
            for i in range(iters):
                t = measure_detect_and_embed(
                    wrapper, iters=1, frame_h=frame_h, frame_w=frame_w
                )
                times_ms.extend(t)
                if (i + 1) % 5 == 0 or (i + 1) == iters:
                    send_log(f"Processed {i+1}/{iters} synthetic frames…")
        mode = "detect_and_embed"

    else:  # 'embed'
        if frames:
            for i, img in enumerate(frames, 1):
                if img is None:
                    continue
                t = measure_embed_only(wrapper, iters=1, override_img=img)
                times_ms.extend(t)
                if i % 5 == 0 or i == len(frames):
                    send_log(f"Processed {i}/{len(frames)} frames…")
        else:
            for i in range(iters):
                t = measure_embed_only(wrapper, iters=1)
                times_ms.extend(t)
                if (i + 1) % 5 == 0 or (i + 1) == iters:
                    send_log(f"Processed {i+1}/{iters} synthetic frames…")
        mode = "embed_only"

    elapsed = time.time() - start
    send_log(f"Completed {len(times_ms)} iterations in {elapsed:.2f}s")

    # Convert latency samples to FPS
    fps_series = [1000.0 / t if t > 0 else float("inf") for t in times_ms]
    mean_fps = float(np.mean(fps_series)) if fps_series else float("nan")

    payload = {
        "model": model_name,
        "mode": mode,
        "dataset": dataset or "synthetic",
        "fps": mean_fps,
        "fps_series": fps_series,
    }

    # --- JSON output consumed by GUI ---
    print(json.dumps(payload))
    sys.stdout.flush()


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
        help="detect=end-to-end pipeline, embed=model-only",
    )
    ap.add_argument(
        "--frame-size",
        default="640x640",
        help="HxW synthetic frame size (default: 640x640)",
    )
    args = ap.parse_args()

    cfg = _resolve_settings()
    model = args.model or cfg.get("model")
    dataset = cfg.get("dataset")  # comes from GUI selection

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

    run(model, args.iters, args.target, h, w, dataset=dataset)


if __name__ == "__main__":
    main()
