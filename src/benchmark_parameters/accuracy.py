import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- Bootstrap sys.path so project root is importable ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from connector import load_model

# -------------------------------
# Args
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name (arcface|facenet|deepface)")
args = parser.parse_args()
print(f"[DEBUG] accuracy.py running with model={args.model}")

# -------------------------------
# Config
# -------------------------------
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
USER1_EMBED_DIR = os.path.join(BASE_DIR, "models", "embeddings", "user1")
USER1_EMBED_FILE = os.path.join(USER1_EMBED_DIR, "embeddings.npy")

SIM_THRESHOLD = 0.5
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "found_matches.txt")

# -------------------------------
# Load user1 embeddings
# -------------------------------
if not os.path.exists(USER1_EMBED_FILE):
    raise FileNotFoundError(f"[ERROR] Missing {USER1_EMBED_FILE}. Run embeddings.py first.")

user1_data = np.load(USER1_EMBED_FILE, allow_pickle=True)
user1_embs = np.stack([entry["embedding"] for entry in user1_data])
print(f"[INFO] Loaded {len(user1_embs)} user1 embeddings.")

# -------------------------------
# Init selected model wrapper
# -------------------------------
wrapper = load_model(args.model)


def extract_embedding(image_path):
    """Get embedding for an image using the wrapper."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Could not read: {image_path}")
        return None

    faces = wrapper.detect_and_embed(img)
    if faces:
        return faces[0]["embedding"]

    # fallback: center crop + embed
    h, w = img.shape[:2]
    min_dim = min(h, w)
    crop = img[(h - min_dim)//2:(h + min_dim)//2,
               (w - min_dim)//2:(w + min_dim)//2]
    return wrapper.embed(crop)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -------------------------------
# Compare dataset images
# -------------------------------
y_true, scores, matches = [], [], []

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

        label = 1 if "user1" in root.lower() else 0
        y_true.append(label)
        scores.append(best_sim)

        if best_sim >= SIM_THRESHOLD:
            print(f"[MATCH] {path} (similarity={best_sim:.3f})")
            matches.append(f"{path} (similarity={best_sim:.3f})")
        else:
            print(f"[NO MATCH] {path} (similarity={best_sim:.3f})")

# -------------------------------
# Save matches
# -------------------------------
if matches:
    with open(OUTPUT_FILE, "w") as f:
        for m in matches:
            f.write(m + "\n")
    print(f"[INFO] Saved {len(matches)} matches to {OUTPUT_FILE}")
else:
    print("[INFO] No matches found, nothing saved.")

# -------------------------------
# Plot results
# -------------------------------
pos = [s for s, y in zip(scores, y_true) if y == 1]
neg = [s for s, y in zip(scores, y_true) if y == 0]

plt.hist(pos, bins=50, alpha=0.6, label="Positive (user1)", color="g")
plt.hist(neg, bins=50, alpha=0.6, label="Negative (others)", color="r")
plt.axvline(SIM_THRESHOLD, color="blue", linestyle="--", label=f"Threshold={SIM_THRESHOLD}")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.legend()
plt.title(f"Face Verification (model={args.model})")
plt.show()
