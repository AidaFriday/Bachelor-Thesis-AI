
#test file
import os
import sys
import numpy as np

# --- make project root importable (same style as your other scripts) ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from connector import load_model  # your central loader


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main():
    lfw_root = r"C:\programming\Datasets\LFW\lfw-deepfunneled"

    # --- pick two images from first identity, one from second identity ---
    # --- automatically pick identities that have enough images ---
    persons = sorted(
        d for d in os.listdir(lfw_root) if os.path.isdir(os.path.join(lfw_root, d))
    )
    if len(persons) < 2:
        raise RuntimeError("Need at least two identities in LFW folder.")

    def jpgs_in(person):
        p = os.path.join(lfw_root, person)
        return sorted(
            f
            for f in os.listdir(p)
            if f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")
        )

    personA, imgsA = None, None
    personB, imgsB = None, None

    for person in persons:
        imgs = jpgs_in(person)

        # first identity with at least 2 images → A
        if personA is None and len(imgs) >= 2:
            personA, imgsA = person, imgs
            continue

        # another identity (different name) with at least 1 image → B
        if (
            personA is not None
            and personB is None
            and person != personA
            and len(imgs) >= 1
        ):
            personB, imgsB = person, imgs
            break

    if personA is None or personB is None:
        raise RuntimeError(
            "Could not find suitable identities: "
            f"personA={personA}, personB={personB}"
        )

    imgA1 = os.path.join(lfw_root, personA, imgsA[0])
    imgA2 = os.path.join(lfw_root, personA, imgsA[1])
    imgB = os.path.join(lfw_root, personB, imgsB[0])

    print("[INFO] Using images:")
    print("   A1:", imgA1)
    print("   A2:", imgA2)
    print("   B :", imgB)

    print("[INFO] loading model: arcface")
    wrapper = load_model("arcface")

    print("[INFO] computing embeddings...")
    embA1 = wrapper.get_embedding(imgA1)
    embA2 = wrapper.get_embedding(imgA2)
    embB = wrapper.get_embedding(imgB)

    print("same person (A1 vs A2):", cosine_similarity(embA1, embA2))
    print("different persons (A1 vs B):", cosine_similarity(embA1, embB))


if __name__ == "__main__":
    main()
