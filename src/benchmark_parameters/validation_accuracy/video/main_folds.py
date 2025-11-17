import os
import platform
import subprocess
from pathlib import Path

# ======================================================================
# 🔧 *** USER CONFIG — SET YOUR EMBEDDING FILE HERE ***
# Use absolute path or just filename if it is inside exports/
USE_EMB_FILE = "facenet_ytf_video_embs_20251117-131203.npz"
# ======================================================================


# Path to logic_ytf_pairs.py (absolute)
SCRIPT = Path(__file__).parent / "logic_ytf_pairs.py"

# Detect OS
os_name = platform.system().lower()
print(f"[run_ytf_pairs] Detected OS: {os_name}")

if os_name == "windows":
    DATASET = r"C:\programming\Datasets\YTF"
    META = r"C:\programming\Datasets\meta_data\meta_and_splits.mat"

    # Path to the REAL exports folder (2 levels up)
    exports_dir = Path(__file__).resolve().parents[2] / "exports"

    # Allow user to specify absolute path OR just filename
    emb_path = Path(USE_EMB_FILE)
    if not emb_path.is_absolute():
        emb_path = exports_dir / USE_EMB_FILE

    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file not found:\n{emb_path}")

    EMBS = str(emb_path)


# === Linux / WSL ===
else:
    DATASET = "/home/aida/Datasets/YTF"
    META = "/home/aida/Datasets/metadata/meta_and_splits.mat"

    emb_path = Path(USE_EMB_FILE)
    if not emb_path.is_absolute():
        emb_path = Path(__file__).parent / USE_EMB_FILE

    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file not found:\n{emb_path}")

    EMBS = str(emb_path)


cmd = [
    "python",
    str(SCRIPT),
    "--model",
    "adaface",
    "--dataset",
    DATASET,
    "--meta",
    META,
    "--embs",
    EMBS,
    "--fold",
    "-1",
]

print("[run_ytf_pairs] Running command:")
print(" ", " ".join(cmd))

subprocess.run(cmd)
