# dataset/LFW.py

import os
import cv2
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# dataset/LFW.py
DATASET_DIR = r"C:\programming\Datasets\LFW\lfw-deepfunneled"


def list_all_images(root_dir, limit=None, shuffle=True, verbose=True):
    """
    Collect images from LFW dataset folder.
    Args:
        root_dir: path to LFW dataset (can be parent or 'lfw-deepfunneled')
        limit: max number of images to load
        shuffle: shuffle before limiting
        verbose: print logs
    Returns:
        List of absolute image paths
    """
    # If user gave parent folder, look inside lfw-deepfunneled
    deep = os.path.join(root_dir, "lfw-deepfunneled")
    if os.path.exists(deep):
        root_dir = deep

    all_images = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append(os.path.join(root, f))

    if shuffle:
        random.shuffle(all_images)

    if limit:
        all_images = all_images[:limit]

    if verbose:
        print(f"[INFO] Found {len(all_images)} images")
        for idx, path in enumerate(all_images, 1):
            print(f" {idx:03d}: {path}")

    return all_images


def load_image(path: str):
    """Read an image from disk in BGR (OpenCV format) with logging."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"[ERROR] Image not found: {path}")
    print(f"[INFO] Loaded image: {path}")
    return img


if __name__ == "__main__":
    # Example: extract 50 images and log them
    images = list_all_images(limit=50, shuffle=True, verbose=True)

    # Test loading the first image
    if images:
        img = load_image(images[0])
        print(f"[INFO] First image shape: {img.shape}")
