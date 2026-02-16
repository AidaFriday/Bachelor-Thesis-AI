import os
import cv2
import random
import sys
import math

# --- Windows-safe stdout setup ---
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    # Fallback for older Python versions
    pass


def _safe_print(msg: str):
    """Print safely across all OS terminals (avoids UnicodeEncodeError on Windows)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "ignore").decode())


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = r"C:\programming\Datasets\YTF"


def list_all_images(
    root_dir,
    limit=None,
    shuffle=True,
    verbose=True,
    success_log=None,
    list_log_limit=0,
):
    """
    Collect video frame images from the YouTube Faces (YTF / aligned_images_DB) dataset.

    Args:
        root_dir: path to YTF dataset (can be parent folder or aligned_images_DB)
        limit: maximum number of frames to load
        shuffle: randomize before truncation
        verbose: print informative logs (counts and optional sample list)
        success_log: if None, follows `verbose`. If True/False, force success/failure
                     summary line on or off.
        list_log_limit: when >0 and verbose=True, print up to this many example paths
                        instead of listing every file. 0 disables per-file listing.
    Returns:
        List of absolute frame image paths
    """
    # If user gave parent folder, check for aligned_images_DB subfolder
    aligned = os.path.join(root_dir, "aligned_images_DB")
    if os.path.exists(aligned):
        root_dir = aligned

    all_images = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".jpg"):
                all_images.append(os.path.join(root, f))

    if shuffle:
        random.shuffle(all_images)

    if limit:
        all_images = all_images[:limit]

    # Decide whether to show the success/warning banner
    if success_log is None:
        success_log = verbose

    if verbose:
        _safe_print(f"[INFO] Found {len(all_images)} frames in YTF dataset")
        if list_log_limit and len(all_images) > 0:
            # show only a small sample to avoid huge logs
            sample = all_images[:list_log_limit]
            for idx, path in enumerate(sample, 1):
                _safe_print(f" {idx:03d}: {path}")
            if len(all_images) > list_log_limit:
                _safe_print(f" ... ({len(all_images) - list_log_limit} more)")

    # Final confirmation banner (now gated)
    if success_log:
        if all_images:
            _safe_print(
                f"\n[SUCCESS] YTF dataset loaded successfully ({len(all_images)} frames found)"
            )
        else:
            _safe_print(f"\n⚠️ [WARNING] No frames found in {root_dir}")

    return all_images


def load_image(path: str):
    """Read a single video frame (image) from disk using OpenCV."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"[ERROR] Frame not found: {path}")
    _safe_print(f"[INFO] Loaded frame: {path}")
    return img


def list_images_by_subset(root_dir, subset="A", limit=None, shuffle=False):
    """
    Divide YTF dataset folder-wise (by person) into 5 subsets (A–E),
    and return up to `limit` image paths from the selected subset.

    Each subset contains complete folders of people,
    so frames of one person never get split across subsets.
    """
    aligned = os.path.join(root_dir, "aligned_images_DB")
    if os.path.exists(aligned):
        root_dir = aligned

    persons = [
        d
        for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    total_persons = len(persons)
    if total_persons == 0:
        raise FileNotFoundError(f"[YTF] No person folders found in {root_dir}")

    subsets = ["A", "B", "C", "D", "E"]
    if subset.upper() not in subsets:
        raise ValueError(f"Invalid subset {subset}. Must be one of {subsets}")

    split_size = math.ceil(total_persons / len(subsets))
    start = subsets.index(subset.upper()) * split_size
    end = min(start + split_size, total_persons)

    selected_persons = persons[start:end]
    _safe_print(
        f"[YTF] Subset {subset.upper()}: persons {start+1}-{end} / {total_persons}"
    )

    all_images = []
    for person in selected_persons:  # loop over each person
        person_dir = os.path.join(root_dir, person)  # e.g., YTF/Aaron_Eckhart
        for root, _, files in os.walk(person_dir):  # loop through all subfolders
            for f in files:  # loop through files
                if f.lower().endswith(".jpg"):  # filter JPGs
                    all_images.append(os.path.join(root, f))  # store full image path

    if limit:
        all_images = all_images[:limit]

    _safe_print(
        f"[YTF] Subset {subset.upper()} → {len(all_images)} frames from {len(selected_persons)} people"
    )
    return all_images


if __name__ == "__main__":
    # Optional quick test — not required for GUI
    pass
