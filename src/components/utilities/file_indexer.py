"""
file_indexer.py — scans the entire project (src/) and indexes all Python files
(excluding __pycache__, hidden folders, and this file itself).
"""

import os
import json
from datetime import datetime

# Auto-detect project root (2 levels above this file)
UTILITIES_DIR = os.path.dirname(os.path.abspath(__file__))
COMPONENTS_DIR = os.path.dirname(UTILITIES_DIR)
PROJECT_ROOT = os.path.dirname(COMPONENTS_DIR)  # <-- src/
INDEX_FILE = os.path.join(UTILITIES_DIR, "file_index.json")


def scan_python_files(root_dir: str):
    """Recursively collect all .py files excluding cache/hidden dirs."""
    python_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [
            d for d in dirnames if not d.startswith(".") and d != "__pycache__"
        ]
        for f in filenames:
            if f.endswith(".py") and not f.startswith("."):
                full_path = os.path.join(dirpath, f)
                rel_path = os.path.relpath(full_path, root_dir)
                python_files.append(rel_path.replace("\\", "/"))
    return sorted(python_files)


def update_file_index():
    """Rebuild file_index.json by scanning the entire src directory."""
    files = scan_python_files(PROJECT_ROOT)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": PROJECT_ROOT,
        "count": len(files),
        "files": files,
    }

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    print(f"[FileIndexer] Indexed {len(files)} Python files → {INDEX_FILE}")
    return payload


def load_file_index():
    """Load existing index, or rebuild if missing."""
    if not os.path.exists(INDEX_FILE):
        return update_file_index()
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return update_file_index()


if __name__ == "__main__":
    update_file_index()
