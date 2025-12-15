

import os
import platform
import subprocess
from pathlib import Path

# ======================================================================
# USER SETTING — PICK ANY NPZ FILE YOU WANT
# Example:
# USE_EMB_FILE = "facenet_ytf_video_embs_20251117-131203.npz"
# USE_EMB_FILE = "arcface_ytf_video_embs_20251117-122752.npz"
# USE_EMB_FILE = "adaface_ytf_video_embs_20251117-163005.npz"

USE_EMB_FILE = "adaface_ytf_video_embs.npz"
# ======================================================================


# Extract the model name automatically:
# "facenet_ytf_video_embs_20251117-131203.npz" → "facenet"
model_name = USE_EMB_FILE.split("_")[0]

# Path to logic_ytf_pairs.py
SCRIPT = Path(__file__).parent / "logic_ytf_pairs.py"

os_name = platform.system().lower()
print(f"[run_ytf_pairs] Detected OS: {os_name}")

if os_name == "windows":

    DATASET = r"C:\programming\Datasets\YTF"
    META = r"C:\programming\Datasets\meta_data\meta_and_splits.mat"

    # real exports folder
    exports_dir = Path(__file__).resolve().parents[2] / "exports"

    emb_path = Path(USE_EMB_FILE)
    if not emb_path.is_absolute():
        emb_path = exports_dir / USE_EMB_FILE

    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file not found:\n{emb_path}")

    EMBS = str(emb_path)

else:
    DATASET = "/home/aida/Datasets/YTF"
    META = "/home/aida/Datasets/meta_data/meta_and_splits.mat"

    # =================================================================
    # NEW: SEARCH FOR THE EMBEDDING FILE IN Test_YTF/*/<USE_EMB_FILE>
    # =================================================================
    EXPORT_ROOT = Path("/home/aida/github/BA_Utilites/BA_tests/Test_YTF")

    found = None
    for sub in EXPORT_ROOT.iterdir():
        if sub.is_dir():
            candidate = sub / USE_EMB_FILE
            if candidate.exists():
                found = candidate
                break

    if found is None:
        raise FileNotFoundError(f"Could not find {USE_EMB_FILE} in {EXPORT_ROOT}")

    EMBS = str(found)
    emb_path = found
    # =================================================================


# ======================================================================
# NEW: CREATE A "folds" DIRECTORY NEXT TO THE FOUND .NPZ FILE
# ======================================================================
output_dir = emb_path.parent / "folds"
output_dir.mkdir(exist_ok=True)

print(f"[run_ytf_pairs] Output folder for this run: {output_dir}")
# ======================================================================


cmd = [
    "python",
    str(SCRIPT),
    "--model",
    model_name,
    "--dataset",
    DATASET,
    "--meta",
    META,
    "--embs",
    EMBS,
    "--fold",
    "-1",
    "--outdir",
    str(output_dir)   # <--- pass output folder
]

print("[run_ytf_pairs] Running command:")
print(" ", " ".join(cmd))

subprocess.run(cmd)
