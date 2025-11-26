# embeddings_extractor.py - not used in testing, i might to delete it later
import sys
import cv2
import numpy as np
from pathlib import Path

# Make /src importable
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[4]  # .../src
sys.path.append(str(SRC_DIR))

from models.wrap_facedetection import FaceDetectorAligner
from connector import load_model


# ---------------------------------------------------------
# Search recursively for a file
# ---------------------------------------------------------
def find_file(filename: str, root_folder: str):
    root = Path(root_folder)
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


# ---------------------------------------------------------
# Extract embedding using correct model logic
# ---------------------------------------------------------
def extract_embedding(model_name: str, img_path: str):
    print(f"[INFO] Loading model: {model_name}")
    wrapper = load_model(model_name)

    # YOLO detector
    detector = FaceDetectorAligner(device="cpu")

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Cannot read: {img_path}")
        return None

    faces = detector.detect(img)
    if not faces:
        print(f"[NO FACE] {img_path}")
        return None

    print(f"[FACE DETECTED] {img_path} → {len(faces)} faces")

    # Model type logic
    name = model_name.lower()
    is_arcface = name == "arcface"
    is_adaface = name == "adaface"
    is_facenet = name in ("facenet", "facenet_onnx")

    emb = None

    try:
        if is_arcface:
            emb = wrapper.get_embedding(img_path)

        elif is_facenet:
            aligned = detector.align_for(img, faces[0]["kps"])
            if aligned is None:
                return None
            emb = wrapper.embed(aligned)

        elif is_adaface:
            emb = wrapper.embed(img)

        else:
            aligned = detector.align_for(img, faces[0]["kps"])
            if aligned is None:
                return None
            emb = wrapper.embed(aligned)

    except Exception as e:
        print(f"[ERROR] Embedding error: {e}")
        return None

    if emb is None:
        return None

    emb = emb.astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-6

    return emb


# ---------------------------------------------------------
# Compute cosine similarity
# ---------------------------------------------------------
def cosine_similarity(a, b):
    return float(np.dot(a, b))


# ---------------------------------------------------------
# Model-specific decision thresholds
# ---------------------------------------------------------
THRESHOLDS = {
    "arcface": 0.30,  # Cosine distance threshold
    "adaface": 0.35,
    "facenet": 0.65,  # Cosine similarity threshold
    "facenet_onnx": 0.65,
}


def get_threshold(model_name: str):
    name = model_name.lower()
    if name in THRESHOLDS:
        return THRESHOLDS[name]
    return 0.50  # default threshold


# ---------------------------------------------------------
# The main function you will use
# ---------------------------------------------------------
def compare_two_faces(model_name, search_root, fname1, fname2):
    print("──────────────────────────────────────────────")
    print(f"[COMPARE] {fname1}   vs   {fname2}")
    print("──────────────────────────────────────────────")

    # --- Find files ---
    f1 = find_file(fname1, search_root)
    f2 = find_file(fname2, search_root)

    if f1 is None:
        print(f"[NOT FOUND] {fname1}")
        return None
    if f2 is None:
        print(f"[NOT FOUND] {fname2}")
        return None

    print(f"[FOUND] 1: {f1}")
    print(f"[FOUND] 2: {f2}")

    # --- Extract embeddings ---
    emb1 = extract_embedding(model_name, str(f1))
    emb2 = extract_embedding(model_name, str(f2))

    if emb1 is None or emb2 is None:
        print("[ERROR] Could not extract embeddings.")
        return None

    # --- Similarity ---
    sim = cosine_similarity(emb1, emb2)
    thresh = get_threshold(model_name)

    print("──────────────────────────────────────────────")
    print(f"[SIMILARITY] {sim:.4f}   (threshold = {thresh})")

    if model_name.lower() in ("facenet", "facenet_onnx"):
        if sim >= thresh:
            print("[MATCH] Same person ✓")
        else:
            print("[NO MATCH] Different persons ✗")
    else:
        # For ArcFace/AdaFace threshold is distance-like
        if sim <= thresh:
            print("[MATCH] Same person ✓")
        else:
            print("[NO MATCH] Different persons ✗")

    print("──────────────────────────────────────────────")
    return sim


# ---------------------------------------------------------
if __name__ == "__main__":
    # Optional: keep CLI mode if you want
    import argparse

    parser = argparse.ArgumentParser(description="Compare two face images")
    parser.add_argument("model", type=str)
    parser.add_argument("root", type=str)
    parser.add_argument("img1", type=str)
    parser.add_argument("img2", type=str)
    args = parser.parse_args()

    compare_two_faces(args.model, args.root, args.img1, args.img2)
