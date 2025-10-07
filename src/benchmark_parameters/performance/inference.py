import argparse, json, os, sys
import numpy as np

# ---- Bootstrap sys.path so project root is importable ----
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from connector import load_model
from latency import measure_detect_and_embed, measure_embed_only

SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")


# ---------------- Helpers ----------------
def send_log(msg, level="info"):
    """Send a log message to GUI as JSON"""
    payload = {"log": msg, "level": level}
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


def _summarize(times_ms):
    if not times_ms:
        return dict(
            avg_ms=float("nan"),
            min_ms=float("nan"),
            max_ms=float("nan"),
            p50_ms=float("nan"),
            p90_ms=float("nan"),
            p95_ms=float("nan"),
        )
    arr = np.array(times_ms, dtype=np.float64)
    return dict(
        avg_ms=float(np.mean(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p90_ms=float(np.percentile(arr, 90)),
        p95_ms=float(np.percentile(arr, 95)),
    )


def run(
    model_name: str,
    iters: int,
    target: str,
    frame_h: int,
    frame_w: int,
    dataset: str = None,
):
    """
    Perform 'iters' inferences and report per-iteration latency in ms.
    """
    wrapper = load_model(model_name)

    send_log(
        f"Running Inference benchmark | Model: {model_name} | "
        f"Dataset: {dataset or 'synthetic'} | Mode: {target}"
    )

    times_ms = []
    if target == "detect":
        mode = "detect_and_embed"
        for i in range(iters):
            t = measure_detect_and_embed(
                wrapper, iters=1, frame_h=frame_h, frame_w=frame_w
            )
            times_ms.extend(t)
            if (i + 1) % 5 == 0 or (i + 1) == iters:
                send_log(f"Progress: {i+1}/{iters} inferences done")
    else:
        mode = "embed_only"
        for i in range(iters):
            t = measure_embed_only(wrapper, iters=1)
            times_ms.extend(t)
            if (i + 1) % 5 == 0 or (i + 1) == iters:
                send_log(f"Progress: {i+1}/{iters} inferences done")

    stats = _summarize(times_ms)

    payload = {
        "kind": "inference",
        "model": model_name,
        "mode": mode,
        "dataset": dataset or "synthetic",
        "count": len(times_ms),
        "times": times_ms,
        **stats,
    }
    print(json.dumps(payload))
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, help="arcface | facenet | insightface")
    ap.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Number of inferences to run (default: 50)",
    )
    ap.add_argument(
        "--target",
        choices=["detect", "embed"],
        default="detect",
        help="detect=end-to-end, embed=model-only",
    )
    ap.add_argument(
        "--frame-size",
        default="640x640",
        help="HxW for detect mode synthetic frames (default: 640x640)",
    )
    args = ap.parse_args()

    cfg = _resolve_settings()
    model = args.model or cfg.get("model")
    dataset = cfg.get("dataset")

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
