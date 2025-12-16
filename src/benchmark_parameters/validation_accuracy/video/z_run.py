import sys
import subprocess
import os

# -----------------------
# MODEL MAPPING
# -----------------------
MODEL_MAP = {
    "1": "facenet",
    "2": "adaface",
    "3": "arcface"
}

# -----------------------
# CONSTANT PATHS
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # current folder

PRECOMPUTE_SCRIPT = os.path.join(BASE_DIR, "ytf_precompute_embeddings.py")
ROC_SCRIPT = os.path.join(BASE_DIR, "logic_roc_ytf_pairs.py")

YTF_DATASET = r"C:\programming\Datasets\YTF"
YTF_META = r"C:\programming\Datasets\meta_data\meta_and_splits.mat"

MAX_FRAMES = "100"


def run(cmd_list):
    print(" ".join(cmd_list))
    subprocess.run(cmd_list)


def run_model(model_name):
    print(f"\n==============================")
    print(f"=== RUNNING MODEL: {model_name}")
    print(f"==============================\n")

    # --- STEP 1: PRECOMPUTE ---
    run([
        "python",
        PRECOMPUTE_SCRIPT,
        "--model", model_name,
        "--dataset", YTF_DATASET,
        "--meta", YTF_META,
        "--max-frames", MAX_FRAMES
    ])

    # --- STEP 2: ROC ---
    run([
        "python",
        ROC_SCRIPT,
        "--model", model_name
    ])


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python z_run.py 1 2 3")
        sys.exit(1)

    for arg in sys.argv[1:]:
        if arg not in MODEL_MAP:
            print(f"Invalid ID '{arg}'. Valid: {list(MODEL_MAP.keys())}")
            continue

        run_model(MODEL_MAP[arg])
