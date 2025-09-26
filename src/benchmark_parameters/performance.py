import argparse
import time
import numpy as np
import os, json, sys

# --- Bootstrap sys.path ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from connector import load_model

SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")


def measure_inference_time(wrapper, model_name: str, iters: int = 50):
    print(f"[DEBUG] Starting performance test for model '{model_name}'")
    h, w = wrapper.input_size
    x = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    # Warm-up
    for _ in range(5):
        _ = wrapper.embed(x)

    start = time.perf_counter()
    for _ in range(iters):
        _ = wrapper.embed(x)
    end = time.perf_counter()

    avg_ms = (end - start) * 1000.0 / iters
    fps = 1000.0 / avg_ms if avg_ms > 0 else float("inf")

    print(f"[RESULT] Model: {model_name}")
    print(f"         Avg inference: {avg_ms:.2f} ms | {fps:.1f} FPS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name (arcface|facenet|deepface)")
    parser.add_argument("--iters", type=int, default=50, help="Number of iterations")
    args = parser.parse_args()

    # If not given via CLI → load from settings.json
    if not args.model:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
            args.model = data.get("model")
    if not args.model:
        print("[ERROR] No model selected in settings.")
        sys.exit(1)

    print(f"[DEBUG] performance.py running with model={args.model}")
    wrapper = load_model(args.model)
    measure_inference_time(wrapper, args.model, iters=args.iters)


if __name__ == "__main__":
    main()
