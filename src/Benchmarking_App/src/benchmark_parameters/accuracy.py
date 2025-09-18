import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis

# -------------------------------
# Config
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src root
DATASET_DIR = os.path.join(BASE_DIR, "dataset")  # adjust if dataset path different
USER1_EMBED_DIR = os.path.join(BASE_DIR, "models", "embeddings", "user1")
USER1_EMBED_FILE = os.path.join(USER1_EMBED_DIR, "embeddings.npy")

# Threshold for cosine similarity (tune this after plotting)
SIM_THRESHOLD = 0.5

# Output text file (next to this script)
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "found_matches.txt")

# -------------------------------
# Load embeddings from user1
# -------------------------------
if not os.path.exists(USER1_EMBED_FILE):
    raise FileNotFoundError(
        f"[ERROR] Missing {USER1_EMBED_FILE}. Run embeddings.py first."
    )

user1_data = np.load(USER1_EMBED_FILE, allow_pickle=True)
user1_embs = np.stack([entry["embedding"] for entry in user1_data])
print(f"[INFO] Loaded {len(user1_embs)} user1 embeddings.")

# -------------------------------
# Init ArcFace
# -------------------------------
MODEL_ROOT = os.path.join(BASE_DIR, "models", "pretrained_models")
app = FaceAnalysis(name="buffalo_l", root=MODEL_ROOT)
app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.3)


def extract_embedding(image_path):
    """Get embedding with fallback mode."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Could not read: {image_path}")
        return None
    faces = app.get(img)
    if len(faces) > 0:
        return faces[0].embedding.astype(np.float32)
    # fallback: resize only
    resized = cv2.resize(img, (112, 112))
    emb = app.models["recognition"].get(resized).astype(np.float32)
    return emb


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -------------------------------
# Compare dataset images
# -------------------------------
y_true, scores = [], []  # for analysis
matches = []  # store successful matches

for root, _, files in os.walk(DATASET_DIR):
    for fname in files:
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        path = os.path.join(root, fname)
        emb = extract_embedding(path)
        if emb is None:
            continue

        sims = [cosine_sim(emb, u) for u in user1_embs]
        best_sim = max(sims)

        # Heuristic: if inside user1 folder → positive label, else negative
        label = 1 if "user1" in root.lower() else 0
        y_true.append(label)
        scores.append(best_sim)

        if best_sim >= SIM_THRESHOLD:
            print(f"[MATCH] {path} (similarity={best_sim:.3f})")
            matches.append(f"{path} (similarity={best_sim:.3f})")
        else:
            print(f"[NO MATCH] {path} (similarity={best_sim:.3f})")

# -------------------------------
# Save matches to txt file
# -------------------------------
if matches:
    with open(OUTPUT_FILE, "w") as f:
        for m in matches:
            f.write(m + "\n")
    print(f"[INFO] Saved {len(matches)} matches to {OUTPUT_FILE}")
else:
    print("[INFO] No matches found, nothing saved.")

# -------------------------------
# Plot similarity distributions
# -------------------------------
pos = [s for s, y in zip(scores, y_true) if y == 1]
neg = [s for s, y in zip(scores, y_true) if y == 0]

plt.hist(pos, bins=50, alpha=0.6, label="Positive (user1)", color="g")
plt.hist(neg, bins=50, alpha=0.6, label="Negative (others)", color="r")
plt.axvline(
    SIM_THRESHOLD, color="blue", linestyle="--", label=f"Threshold={SIM_THRESHOLD}"
)
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.legend()
plt.title("Face Verification: Similarity Distributions")
plt.show()
