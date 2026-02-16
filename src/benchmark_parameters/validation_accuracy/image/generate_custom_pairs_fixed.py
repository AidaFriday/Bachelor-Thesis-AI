# generate_custom_pairs_fixed.py
# EXACT: 1080 POS, 1080 NEG, 10 FOLDS, NO DUPLICATES

import os
import random
import sys
from itertools import combinations, product

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)

DATASET_PATH = r"C:/programming/Datasets/CUSTOM_DATASET_ORG"
OUTPUT_FILE = "pairs_custom.txt"

NUM_FOLDS = 10
TOTAL_POS = 1080
TOTAL_NEG = 1080

POS_PER_FOLD = TOTAL_POS // NUM_FOLDS  # 108
NEG_PER_FOLD = TOTAL_NEG // NUM_FOLDS  # 108

random.seed(42)  # reproducible


# --------------------------------------------------------------
# Load dataset: { person: [img1.jpg, img2.jpg, ...] }
# --------------------------------------------------------------
def load_dataset(dataset_path):
    people = {}

    for person in sorted(os.listdir(dataset_path)):
        folder = os.path.join(dataset_path, person)
        if not os.path.isdir(folder):
            continue

        images = [
            f
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(images) >= 2:
            people[person] = sorted(images)

    return people


# --------------------------------------------------------------
# Create ALL POSSIBLE POSITIVE PAIRS
# --------------------------------------------------------------
def build_positive_pairs(people):
    pos = []

    for person, imgs in people.items():
        for img1, img2 in combinations(imgs, 2):
            pos.append((person, img1, img2))

    random.shuffle(pos)
    return pos


# --------------------------------------------------------------
# Create ALL POSSIBLE NEGATIVE PAIRS
# --------------------------------------------------------------
def build_negative_pairs(people):
    persons = list(people.keys())
    neg = []

    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            p1, p2 = persons[i], persons[j]
            for img1, img2 in product(people[p1], people[p2]):
                neg.append((p1, img1, p2, img2))

    random.shuffle(neg)
    return neg


# --------------------------------------------------------------
# Split into folds WITHOUT repetition
# --------------------------------------------------------------
def build_folds(pos_pairs, neg_pairs):
    required_pos = TOTAL_POS
    required_neg = TOTAL_NEG

    if len(pos_pairs) < required_pos:
        raise ValueError(
            f"Not enough positive pairs: need {required_pos}, have {len(pos_pairs)}"
        )

    if len(neg_pairs) < required_neg:
        raise ValueError(
            f"Not enough negative pairs: need {required_neg}, have {len(neg_pairs)}"
        )

    # take only what we need
    pos_pairs = pos_pairs[:required_pos]
    neg_pairs = neg_pairs[:required_neg]

    folds = []

    for f in range(NUM_FOLDS):
        start_p = f * POS_PER_FOLD
        end_p = start_p + POS_PER_FOLD

        start_n = f * NEG_PER_FOLD
        end_n = start_n + NEG_PER_FOLD

        fold_pos = pos_pairs[start_p:end_p]
        fold_neg = neg_pairs[start_n:end_n]

        folds.append((fold_pos, fold_neg))

    return folds


# --------------------------------------------------------------
# Write pairs file (LFW-style format)
# --------------------------------------------------------------
def write_pairs(folds, out_file):
    with open(out_file, "w") as f:
        # header: <num_folds> <pairs_per_fold>
        f.write(f"{NUM_FOLDS} {POS_PER_FOLD}\n")

        for positives, negatives in folds:

            # positive pairs
            for p, i1, i2 in positives:
                f.write(f"{p} {i1} {i2}\n")

            # negative pairs
            for p1, i1, p2, i2 in negatives:
                f.write(f"{p1} {i1} {p2} {i2}\n")

    print(f"[OK] Saved pairs file: {out_file}")
    print(f"[INFO] Total positive pairs: {TOTAL_POS}")
    print(f"[INFO] Total negative pairs: {TOTAL_NEG}")
    print(f"[INFO] Folds: {NUM_FOLDS} (each: {POS_PER_FOLD} pos + {NEG_PER_FOLD} neg)")


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
if __name__ == "__main__":
    people = load_dataset(DATASET_PATH)

    print(f"[INFO] Loaded {len(people)} identities")

    pos_pairs = build_positive_pairs(people)
    neg_pairs = build_negative_pairs(people)

    print(f"[INFO] Positive pairs available: {len(pos_pairs)}")
    print(f"[INFO] Negative pairs available: {len(neg_pairs)}")

    folds = build_folds(pos_pairs, neg_pairs)
    write_pairs(folds, OUTPUT_FILE)
