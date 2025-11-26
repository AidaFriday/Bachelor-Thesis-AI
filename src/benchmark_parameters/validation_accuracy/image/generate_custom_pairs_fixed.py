#generate_custom_pairs_fixed.py
import os
import random
import sys

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)

DATASET_PATH = r"C:/programming/Datasets/CUSTOM_DATASET_CLEAN"
OUTPUT_FILE = "pairs_custom.txt"

NUM_FOLDS = 3
PAIRS_PER_FOLD = 10  # positive + negative

random.seed(42)   # IMPORTANT for reproducibility


def load_dataset(dataset_path):
    people = {}

    for person in sorted(os.listdir(dataset_path)):
        folder = os.path.join(dataset_path, person)
        if not os.path.isdir(folder):
            continue

        images = [f for f in os.listdir(folder)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if len(images) >= 2:
            people[person] = sorted(images)

    return people


def create_pairs(people):
    folds = []
    persons = list(people.keys())

    for fold in range(NUM_FOLDS):

        positives = []
        negatives = []

        # Positive pairs
        for person in persons:
            if len(positives) >= PAIRS_PER_FOLD:
                break

            imgs = people[person]
            if len(imgs) < 2:
                continue

            img1, img2 = random.sample(imgs, 2)
            positives.append((person, img1, img2))

        # Negative pairs
        while len(negatives) < PAIRS_PER_FOLD:
            p1, p2 = random.sample(persons, 2)
            img1 = random.choice(people[p1])
            img2 = random.choice(people[p2])
            negatives.append((p1, img1, p2, img2))

        folds.append((positives, negatives))

    return folds


def write_pairs(folds, out_file):
    with open(out_file, "w") as f:
        f.write(f"{NUM_FOLDS} {PAIRS_PER_FOLD}\n")

        for positives, negatives in folds:
            for p, img1, img2 in positives:
                f.write(f"{p} {img1} {img2}\n")

            for p1, img1, p2, img2 in negatives:
                f.write(f"{p1} {img1} {p2} {img2}\n")

    print(f"[OK] Saved pairs file: {out_file}")


if __name__ == "__main__":
    people = load_dataset(DATASET_PATH)
    folds = create_pairs(people)
    write_pairs(folds, OUTPUT_FILE)
