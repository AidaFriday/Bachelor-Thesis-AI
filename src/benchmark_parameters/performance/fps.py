import argparse, json, os, sys, time
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
from dataset import LFW

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


def _random_frame(h=640, w=640):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def measure_once(wrapper, mode="detect", frame=None):
    """Measure latency for one run (detect+embed or embed only)."""
    _cuda_synchronize_if_needed()
    t0 = time.perf_counter()
    if mode == "detect":
        _ = wrapper.detect_and_embed(frame)
    else:
        _ = wrapper.embed(frame)
    _cuda_synchronize_if_needed()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0  # ms


def run(model_name, iters, target, frame_h, frame_w, dataset=None):
    wrapper = load_model(model_name)

    send_log(
        f"Running FPS benchmark | Model: {model_name} | "
        f"Dataset: {dataset or 'synthetic'} | Mode: {target}"
    )

    frames = []
    image_map = {}

    # ✅ Load dataset if provided
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

    times_ms = []
    # START total runtime timer
    start = time.time()

    for i in range(iters):
        # pick frame from dataset or synthetic
        if frames:
            frame = frames[i % len(frames)]
        else:
            frame = _random_frame(frame_h, frame_w)

        t = measure_once(wrapper, mode=target, frame=frame)
        times_ms.append(t)

        if (i + 1) % 5 == 0 or (i + 1) == iters:
            send_log(f"Processed {i+1}/{iters} frames…")

    # END total runtime timer
    elapsed = time.time() - start
    send_log(f"Completed {iters} iterations in {elapsed:.2f}s", level="result")

    # Convert latency → FPS
    fps_series = [1000.0 / t if t > 0 else float("inf") for t in times_ms]
    mean_fps = float(np.mean(fps_series)) if fps_series else float("nan")

    # log
    send_log(
        f"Average FPS on {dataset or 'synthetic'} ({len(fps_series)} frames): {mean_fps:.2f} FPS",
        level="result",
    )

    payload = {
        "kind": "fps",
        "model": model_name,
        "mode": "detect_and_embed" if target == "detect" else "embed_only",
        "dataset": dataset or "synthetic",
        "fps": mean_fps,
        "fps_series": fps_series,
    }

    if dataset and frames:
        mapping_file = os.path.join(PROJECT_ROOT, "fps_index_map.json")
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
    parser.add_argument("--iters", type=int, default=50, help="Number of iterations")
    parser.add_argument("--target", choices=["detect", "embed"], default="detect")
    parser.add_argument("--frame-size", type=str, default="640x640", help="HxW size")
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

    run(model, args.iters, args.target, h, w, dataset=dataset)


if __name__ == "__main__":
    main()
