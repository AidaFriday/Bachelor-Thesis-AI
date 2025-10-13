import argparse, json, os, sys, time
import numpy as np
import cv2

# ---- Bootstrap sys.path so connector and dataset are importable ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONNECTOR_PARENT = os.path.dirname(
    os.path.dirname(CURRENT_DIR)
)  # one level up (benchmark_parameters)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))  # src

for p in [CONNECTOR_PARENT, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from connector import load_model
from dataset import LFW, YTF

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


def _ytf_loaded_subset_summary(root_dir: str, image_paths):
    """
    Summarize the loaded YTF subset (subjects/videos/frames).
    Assumes layout: subject / video / aligned_detect_*_{frame}.jpg
    """
    subjects = set()
    videos = set()
    for p in image_paths:
        rel = os.path.relpath(p, root_dir)
        parts = rel.split(os.sep)
        if len(parts) >= 3:
            subjects.add(parts[0])
            videos.add(os.path.join(parts[0], parts[1]))
    return len(subjects), len(videos), len(image_paths)


def run(model_name, iters, target, frame_h, frame_w, dataset=None):
    wrapper = load_model(model_name)

    send_log(
        f"Running FPS benchmark | Model: {model_name} | "
        f"Dataset: {dataset or 'synthetic'} | Mode: {target}"
    )

    frames = []
    image_map = {}

    # -------- Load dataset automatically (LFW or YTF) --------
    images = []
    dataset_name = "synthetic"
    selected_subjects_env = os.getenv("YTF_SELECTED_SUBJECTS")

    # ✅ Parse selected subjects (passed from GUI)
    if selected_subjects_env:
        selected_subjects = set(
            s.strip() for s in selected_subjects_env.split(",") if s.strip()
        )
        send_log(
            f"Filtering dataset to selected subjects: {', '.join(selected_subjects)}",
            level="info",
        )
    else:
        selected_subjects = None

    if dataset:
        dl = dataset.lower()
        if ("ytf" in dl) or ("aligned_images_db" in dl):
            dataset_name = "YTF (aligned)"

            all_images = YTF.list_all_images(
                root_dir=dataset, limit=None, shuffle=False, verbose=False
            )
            all_images.sort()  # enforce deterministic order across OS

            # ✅ Filter to selected subjects if provided
            if selected_subjects:
                images = [
                    p
                    for p in all_images
                    if os.path.basename(os.path.dirname(os.path.dirname(p)))
                    in selected_subjects
                ]
                if not images:
                    send_log("⚠️ No images found for selected subjects", level="error")
                    return
            else:
                images = all_images  # use everything

            # ✅ Automatically set iteration count to number of found images
            iters = len(images)

            s, v, f = _ytf_loaded_subset_summary(dataset, images)
            send_log(
                f"[YTF] Loaded subset: {s} subjects, {v} videos, {f} frames",
                level="result",
            )

        else:
            dataset_name = "LFW"
            images = LFW.list_all_images(
                root_dir=dataset, limit=None, shuffle=True, verbose=False
            )
            iters = len(images)

    # Load frames into memory (works for both LFW and YTF aligned frames)
    for idx, path in enumerate(images, 1):
        img = cv2.imread(path)
        if img is not None:
            frames.append(img)
            image_map[idx] = os.path.basename(path)
            send_log(f"[{idx:03d}] Loaded dataset image: {path}")

    # -------- Measure --------
    num_runs = int(os.getenv("YTF_RUNS", "1"))
    send_log(f"[CONFIG] Running each dataset {num_runs} time(s)")

    all_run_fps = []
    all_run_series = []

    for run_idx in range(num_runs):
        send_log(f"--- Run {run_idx+1}/{num_runs} ---")
        times_ms = []
        start = time.time()

        for i in range(iters):
            frame = (
                frames[i % len(frames)] if frames else _random_frame(frame_h, frame_w)
            )
            t = measure_once(wrapper, mode=target, frame=frame)
            times_ms.append(t)

            if (i + 1) % 5 == 0 or (i + 1) == iters:
                send_log(f"Processed {i+1}/{iters} frames (run {run_idx+1})…")

        elapsed = time.time() - start
        fps_series = [1000.0 / t if t > 0 else float("inf") for t in times_ms]
        mean_fps = float(np.mean(fps_series)) if fps_series else float("nan")

        all_run_fps.append(mean_fps)
        all_run_series.append(fps_series)  # ✅ store each run’s FPS list

        send_log(
            f"[Run {run_idx+1}] Completed {iters} frames in {elapsed:.2f}s → {mean_fps:.2f} FPS",
            level="result",
        )

    # ---- Final summary ----
    if num_runs > 1:
        avg_fps = sum(all_run_fps) / len(all_run_fps)
        send_log(
            f"[RESULT] Average FPS over {num_runs} runs: {avg_fps:.2f}", level="result"
        )
    else:
        avg_fps = all_run_fps[0]

    payload = {
        "kind": "fps",
        "model": model_name,
        "mode": "detect_and_embed" if target == "detect" else "embed_only",
        "dataset": dataset_name if dataset else "synthetic",
        "fps": avg_fps,
        "runs": all_run_fps,
        "fps_series_all": all_run_series,  # ✅ now each run’s series is unique
    }

    # ---- Build and save run summary report ----
    if dataset and frames:
        report = {"runs": []}

        for run_idx, fps_series in enumerate(all_run_series):
            if not fps_series:
                continue

            min_idx = int(np.argmin(fps_series))
            max_idx = int(np.argmax(fps_series))
            min_fps = float(fps_series[min_idx])
            max_fps = float(fps_series[max_idx])
            avg_fps = float(all_run_fps[run_idx])

            min_file = image_map.get(min_idx + 1, f"frame_{min_idx+1}")
            max_file = image_map.get(max_idx + 1, f"frame_{max_idx+1}")

            report["runs"].append(
                {
                    "run": run_idx + 1,
                    "min_fps": round(min_fps, 2),
                    "max_fps": round(max_fps, 2),
                    "avg_fps": round(avg_fps, 2),
                    "min_file": min_file,
                    "max_file": max_file,
                }
            )

        report_file = os.path.join(PROJECT_ROOT, "fps_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=4)
        send_log(f"Saved per-run summary report to {report_file}")

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
    parser.add_argument(
        "--dataset", type=str, default=None, help="Path to dataset (optional)"
    )
    args = parser.parse_args()

    cfg = _resolve_settings()
    model = args.model or cfg.get("model")
    # ✅ Prefer CLI path if provided; else fall back to settings.json
    dataset = args.dataset or cfg.get("dataset")

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
