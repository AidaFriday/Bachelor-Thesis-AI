import os
import cv2
import json
import time
import random
import numpy as np
from tqdm import tqdm
import sys, os

# Add the "src" directory to Python path dynamically

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from connector import load_model


def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings."""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def generate_pairs(dataset_path, max_pairs=600):
    """Generate same/different image pairs from dataset folders."""
    people = [
        p
        for p in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, p))
    ]
    all_pairs = []

    # Positive pairs (keep only if at least 2 images)
    valid_people = []
    for person in people:
        imgs = [
            os.path.join(dataset_path, person, f)
            for f in os.listdir(os.path.join(dataset_path, person))
            if f.lower().endswith((".jpg", ".png"))
        ]
        if len(imgs) >= 2:
            valid_people.append(person)
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    all_pairs.append((imgs[i], imgs[j], 1))

    # Fallback if no positive pairs (too many single-image folders)
    if not all_pairs:
        print("[WARN] No multi-image identities found; generating only negative pairs")
        people_with_imgs = [
            (
                p,
                [
                    os.path.join(dataset_path, p, f)
                    for f in os.listdir(os.path.join(dataset_path, p))
                    if f.lower().endswith((".jpg", ".png"))
                ],
            )
            for p in people
        ]
        people_with_imgs = [(p, imgs) for p, imgs in people_with_imgs if imgs]
        while len(all_pairs) < max_pairs and len(people_with_imgs) >= 2:
            p1, p2 = random.sample(people_with_imgs, 2)
            img1 = random.choice(p1[1])
            img2 = random.choice(p2[1])
            all_pairs.append((img1, img2, 0))

    random.shuffle(all_pairs)
    print(f"[DEBUG] Generated {len(all_pairs)} pairs from {len(people)} identities")
    return all_pairs[:max_pairs]


def find_best_threshold(sims, labels):
    thresholds = np.linspace(-1, 1, 200)
    best_acc, best_t = 0, 0
    labels = np.array(labels)
    sims = np.array(sims)
    for t in thresholds:
        preds = (sims > t).astype(int)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_acc, best_t


def run_logic(model_path, iters=300, frame_h=None, frame_w=None, dataset_path=None):
    if dataset_path is None:
        dataset_path = model_path

    # --- Auto-fix for common dataset folder structure ---
    if os.path.isdir(os.path.join(dataset_path, "lfw-deepfunneled")):
        dataset_path = os.path.join(dataset_path, "lfw-deepfunneled")

    wrapper = load_model(model_path)
    model_name = getattr(wrapper, "name", os.path.basename(model_path))

    print(f"[DEBUG] run_logic() called with iters={iters}, dataset_path={dataset_path}")

    pairs = generate_pairs(dataset_path, max_pairs=iters)
    sims, labels = [], []
    start_time = time.time()

    # debug print block
    print(f"[INFO] Running validation on {len(pairs)} pairs from {dataset_path}")
    print("Example pairs:")
    for i in range(min(3, len(pairs))):
        print(" ", pairs[i])

    for img1, img2, label in tqdm(pairs, desc="Validating", ncols=80):
        img1 = os.path.normpath(img1)
        img2 = os.path.normpath(img2)
        img_a = cv2.imread(img1)
        img_b = cv2.imread(img2)
        if img_a is None or img_b is None:
            print(f"[WARN] Skipping unreadable pair:\n  {img1}\n  {img2}")
            continue

        emb1 = wrapper.embed(img_a)
        emb2 = wrapper.embed(img_b)
        sims.append(cosine_similarity(emb1, emb2))
        labels.append(label)

    acc, best_t = find_best_threshold(sims, labels)
    elapsed = time.time() - start_time

    result = {
        "kind": "accuracy_image",
        "dataset": os.path.basename(dataset_path),
        "model": model_name,
        "num_pairs": len(pairs),
        "accuracy": round(float(acc), 5),
        "threshold": round(float(best_t), 3),
        "elapsed_sec": round(float(elapsed), 2),
    }
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--iters", type=int, default=300)
    args = parser.parse_args()
    run_logic(args.model_path, args.dataset_path, args.iters)
