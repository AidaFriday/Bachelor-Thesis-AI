# generate_custom_pairs.py
import os
import sys
import random

# FIX: always add project 'src' root to Python import path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)

# ----------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------
DATASET_PATH = r"C:/programming/Datasets/CUSTOM_DATASET"
OUTPUT_FILE = "pairs_custom.txt"

NUM_FOLDS = 3
PAIRS_PER_FOLD = 10  # 10 positive + 10 negative per fold


# ----------------------------------------------------------
# HELPER — extract all people and images
# ----------------------------------------------------------
def load_dataset(dataset_path):
    people = {}
    for name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, name)
        if not os.path.isdir(person_dir):
            continue

        images = [f for f in os.listdir(person_dir) if f.lower().endswith(".jpeg")]

        if len(images) >= 2:  # must have at least 2 images for positive pairs
            people[name] = sorted(images)

    return people


# ----------------------------------------------------------
# BUILD PAIRS
# ----------------------------------------------------------
def create_pairs(people):
    folds = []

    for fold in range(NUM_FOLDS):
        positives = []
        negatives = []

        # ---- positive pairs ----
        for person, images in people.items():
            if len(positives) >= PAIRS_PER_FOLD:
                break

            if len(images) < 2:
                continue

            i1, i2 = random.sample(images, 2)
            n1 = int(i1.split("_")[1].split(".")[0])  # extract 3-digit index
            n2 = int(i2.split("_")[1].split(".")[0])

            positives.append((person, n1, n2))

        # ---- negative pairs ----
        names = list(people.keys())
        while len(negatives) < PAIRS_PER_FOLD:
            p1, p2 = random.sample(names, 2)
            img1 = random.choice(people[p1])
            img2 = random.choice(people[p2])

            n1 = int(img1.split("_")[1].split(".")[0])
            n2 = int(img2.split("_")[1].split(".")[0])

            negatives.append((p1, n1, p2, n2))

        folds.append((positives, negatives))

    return folds


# ----------------------------------------------------------
# WRITE LFW-STYLE PAIRS FILE
# ----------------------------------------------------------
def write_pairs_file(folds, out_file):
    with open(out_file, "w") as f:
        f.write(f"{NUM_FOLDS} {PAIRS_PER_FOLD}\n")

        for positives, negatives in folds:
            for p in positives:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
            for n in negatives:
                f.write(f"{n[0]} {n[1]} {n[2]} {n[3]}\n")

    print(f"[OK] Saved pairs file: {out_file}")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
if __name__ == "__main__":
    people = load_dataset(DATASET_PATH)
    print("[INFO] People found:", list(people.keys()))

    folds = create_pairs(people)
    write_pairs_file(folds, OUTPUT_FILE)
