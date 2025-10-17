# validation_accuracy.py
import argparse
import json
import os
import sys

# dynamic import loader same as latency.py
try:
    from components.utilities.file_indexer import load_file_index
except ModuleNotFoundError:
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "../../components/utilities")
    )
    from file_indexer import load_file_index

index_data = load_file_index()
PROJECT_ROOT = index_data["root"]
for rel_path in index_data["files"]:
    dir_path = os.path.join(PROJECT_ROOT, os.path.dirname(rel_path))
    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)

SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")


def send_log(msg, level="info"):
    print(json.dumps({"log": msg, "level": level}))
    sys.stdout.flush()


def _resolve_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def run(model_name, iters, frame_h, frame_w, dataset=None):
    """Delegates execution to image or video VA logic."""
    dataset = dataset or ""
    dl = dataset.lower()

    if "lfw" in dl:
        from benchmark_parameters.validation_accuracy.logic_dataset_image_va import (
            run_logic as logic_run,
        )

        send_log(f"[accuracy] Using IMAGE dataset logic for {dataset}")
    elif "ytf" in dl or "aligned_images_db" in dl or "video" in dl:
        from benchmark_parameters.validation_accuracy.logic_dataset_video_va import (
            run_logic as logic_run,
        )

        send_log(f"[accuracy] Using VIDEO dataset logic for {dataset}")
    else:
        from benchmark_parameters.validation_accuracy.logic_dataset_image_va import (
            run_logic as logic_run,
        )

        send_log(f"[accuracy] Defaulting to IMAGE dataset logic (no match)")

    logic_run(model_name, iters, frame_h, frame_w, dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--frame-size", type=str, default="640x640")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    cfg = _resolve_settings()
    model = args.model or cfg.get("model")
    dataset = args.dataset or cfg.get("dataset")

    if not model:
        print(json.dumps({"error": "No model selected"}))
        sys.exit(1)

    try:
        h, w = [int(x) for x in args.frame_size.lower().split("x")]
    except Exception:
        h, w = 640, 640

    run(model, args.iters, h, w, dataset)


if __name__ == "__main__":
    main()
