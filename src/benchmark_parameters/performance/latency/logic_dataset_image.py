# ==== logic_dataset_image.py ====
import json, sys, time, numpy as np, os, cv2
from connector import load_model

try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def send_log(msg, level="info"):
    print(json.dumps({"log": msg, "level": level}))
    sys.stdout.flush()


def _cuda_sync():
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_once(wrapper, frame):
    _cuda_sync()
    t0 = time.perf_counter()
    _ = wrapper.embed(frame)
    _cuda_sync()
    return (time.perf_counter() - t0) * 1000.0


def run_logic(model_name, iters, frame_h, frame_w, dataset):
    wrapper = load_model(model_name)

    start_person = os.getenv("LFW_START_PERSON", "")
    img_limit = int(os.getenv("LFW_IMAGE_COUNT", "0"))
    num_runs = int(os.getenv("LFW_RUNS", "1"))

    if not os.path.isdir(dataset):
        send_log(f"❌ Invalid dataset path: {dataset}", "error")
        return

    # ---- Step 1: collect all people in alphabetical order ----
    people = sorted(
        [d for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))]
    )
    if not people:
        send_log("❌ No person folders found", "error")
        return

    # ---- Step 2: find start index ----
    try:
        start_idx = people.index(start_person)
    except ValueError:
        start_idx = 0
        send_log(f"⚠️ Person '{start_person}' not found. Starting from '{people[0]}'")

    # ---- Step 3: gather image paths ----
    image_paths = []
    for person in people[start_idx:]:
        person_dir = os.path.join(dataset, person)
        imgs = sorted(
            [
                os.path.join(person_dir, f)
                for f in os.listdir(person_dir)
                if f.lower().endswith(".jpg")
            ]
        )
        image_paths.extend(imgs)
        if img_limit and len(image_paths) >= img_limit:
            break

    if not image_paths:
        send_log("❌ No images found in selected range", "error")
        return

    # clip to limit
    if img_limit and len(image_paths) > img_limit:
        image_paths = image_paths[:img_limit]

    send_log(
        f"[CONFIG] Start='{start_person}', Count={len(image_paths)}, Runs={num_runs}"
    )

    # ---- Warmup ----
    first_frame = cv2.imread(image_paths[0])
    if first_frame is None:
        send_log("❌ Could not read first image", "error")
        return
    _ = wrapper.embed(first_frame)
    _cuda_sync()

    # ---- Runs ----
    all_runs = []
    avg_runs = []
    for r in range(num_runs):
        send_log(f"--- Run {r+1}/{num_runs} ---")
        latencies = []
        for i, img_path in enumerate(image_paths):
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            t_ms = measure_once(wrapper, frame)
            latencies.append(t_ms)
            if (i + 1) % 5 == 0:
                send_log(f"Processed {i+1}/{len(image_paths)} images")

        if not latencies:
            send_log(f"⚠️ Run {r+1} had no valid images", "warn")
            continue

        avg_ms = float(np.mean(latencies))
        avg_runs.append(avg_ms)
        all_runs.append(latencies)

        send_log(f"[Run {r+1}] Avg={avg_ms:.2f} ms, {1000/avg_ms:.2f} FPS")

    if not all_runs:
        send_log("❌ All runs failed", "error")
        return

    overall_avg = float(np.mean(avg_runs))
    send_log(f"✅ Overall Avg Latency = {overall_avg:.2f} ms", "result")

    payload = {
        "kind": "latency_image",
        "dataset": dataset,
        "start_person": start_person,
        "num_images": len(image_paths),
        "num_runs": num_runs,
        "avg_latency_ms": overall_avg,
        "latency_series_all": all_runs,
        "image_paths": image_paths,
    }
    print(json.dumps(payload))
    sys.stdout.flush()
