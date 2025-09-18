import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Path to pretrained ArcFace buffalo_l
MODEL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "pretrained_models")
)

# Init ArcFace
app = FaceAnalysis(name="buffalo_l", root=MODEL_ROOT)
app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.3)  # lower threshold


def extract_embedding(image_path):
    """Extract ArcFace embedding. Falls back to resize-only mode if no face detected."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Could not read: {image_path}")
        return None

    faces = app.get(img)
    if len(faces) > 0:
        return faces[0].embedding.astype(np.float32)

    # fallback: resize whole image
    print(f"[FALLBACK] No face detected in {image_path}, using resize-only mode")
    resized = cv2.resize(img, (112, 112))
    # ArcFace recognition model expects BGR float32 input
    emb = app.models["recognition"].get(resized).astype(np.float32)
    return emb


def process_directory(base_dir):
    """Process each user folder and save embeddings.npy inside it."""
    for person in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person)
        if not os.path.isdir(person_dir):
            continue  # skip files

        embeddings = []
        for fname in os.listdir(person_dir):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            img_path = os.path.join(person_dir, fname)
            emb = extract_embedding(img_path)
            if emb is not None:
                embeddings.append({"file": fname, "embedding": emb})

        if embeddings:
            save_path = os.path.join(person_dir, "embeddings.npy")
            np.save(save_path, embeddings)
            print(f"[OK] {person}: saved {len(embeddings)} embeddings -> {save_path}")
        else:
            print(f"[SKIP] {person}: no embeddings generated")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)  # current folder: embeddings/
    process_directory(BASE_DIR)
