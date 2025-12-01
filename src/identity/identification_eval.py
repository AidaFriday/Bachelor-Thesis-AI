import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from connector import load_model
from identity.face_database import FaceEmbeddingDB


# ------------------------- Utilities -------------------------


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def list_identities(dataset_root: str):
    dataset_root = Path(dataset_root)
    persons = [p for p in dataset_root.iterdir() if p.is_dir()]
    return sorted(persons)


# ------------------------- Closed-set ID -------------------------


def evaluate_closed_set(model_wrapper, db: FaceEmbeddingDB, dataset_root: str):
    """
    Closed-set Rank-1 identification and optional CMC.
    For each identity: each image is treated as a probe, DB contains mean embeddings.
    """
    persons = list_identities(dataset_root)

    correct_rank1 = 0
    total = 0

    rank_positions = []  # For CMC curve

    for person_dir in persons:
        name = person_dir.name
        img_files = [
            p
            for p in person_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        for img_path in img_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Detect and align (ArcFace special case)
            if hasattr(model_wrapper, "name") and model_wrapper.name == "arcface":
                faces = model_wrapper.app.get(img)
                if not faces:
                    continue
                emb = faces[0].embedding.astype(np.float32)
            else:
                faces = model_wrapper.detector.detect(img)
                if not faces:
                    continue
                aligned = model_wrapper.detector.align_for(img, faces[0]["kps"])
                if aligned is None:
                    continue
                emb = model_wrapper.embed(aligned)

            if emb is None:
                continue

            # Compare to all DB embeddings
            sims = [
                cosine_similarity(emb, db.embeddings[i]) for i in range(len(db.names))
            ]

            order = np.argsort(sims)[::-1]
            best_idx = order[0]
            predicted_name = db.names[best_idx]

            total += 1
            if predicted_name == name:
                correct_rank1 += 1

            # For CMC
            true_pos = list(order).index(db.names.index(name))
            rank_positions.append(true_pos)

    rank1 = correct_rank1 / total if total > 0 else 0.0

    # CMC: Compute cumulative probability up to rank-K
    max_rank = max(rank_positions) + 1 if rank_positions else 1
    cmc = np.zeros(max_rank, dtype=np.float32)
    for r in rank_positions:
        cmc[r] += 1
    cmc = np.cumsum(cmc) / total if total > 0 else cmc

    return {
        "rank1": float(rank1),
        "cmc": cmc.tolist(),
    }


# ------------------------- Open-set ID -------------------------


def compute_open_set_metrics(sim_vectors, labels, thresholds):
    """
    sim_vectors[i] = max similarity of probe i against DB identities.
    labels[i] = 1 if probe is known, 0 if unknown.
    """
    tpir_at_fpir = {}

    # Sweeping fpr targets
    targets = [0.01, 0.001]

    for target in targets:
        best_tpir = 0.0
        thr_at_target = thresholds[0]

        for t in thresholds:
            preds = (sim_vectors >= t).astype(int)

            # FPIR = false positive identification rate
            fp = np.sum((preds == 1) & (labels == 0))
            tn = np.sum((preds == 0) & (labels == 0))
            fpir = fp / (fp + tn + 1e-9)

            if fpir <= target:
                tp = np.sum((preds == 1) & (labels == 1))
                fn = np.sum((preds == 0) & (labels == 1))
                tpir = tp / (tp + fn + 1e-9)

                if tpir > best_tpir:
                    best_tpir = tpir
                    thr_at_target = t

        tpir_at_fpir[target] = {
            "threshold": float(thr_at_target),
            "tpir": float(best_tpir),
        }

    return tpir_at_fpir


def evaluate_open_set(model_wrapper, db: FaceEmbeddingDB, dataset_root: str):
    """
    For open-set:
    - Known probes: each image from identities in CUSTOM_DATASET
    - Unknown probes: images from a synthetic 'unknown' split:
      Any identity not in DB or random distractors (here: 20% per-person reserved)
    """

    persons = list_identities(dataset_root)

    sim_vecs = []
    labels = []

    for person_dir in persons:
        name = person_dir.name
        img_files = [
            p
            for p in person_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        # We'll treat LAST 20% of images as "unknown" for open-set simulation
        split = max(1, int(len(img_files) * 0.2))
        known_imgs = img_files[:-split]
        unknown_imgs = img_files[-split:]

        # Known probes
        for img_path in known_imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Extract emb
            emb = None
            if model_wrapper.name == "arcface":
                faces = model_wrapper.app.get(img)
                if faces:
                    emb = faces[0].embedding.astype(np.float32)
            else:
                faces = model_wrapper.detector.detect(img)
                if faces:
                    aligned = model_wrapper.detector.align_for(img, faces[0]["kps"])
                    if aligned is not None:
                        emb = model_wrapper.embed(aligned)

            if emb is None:
                continue

            sims = [
                cosine_similarity(emb, db.embeddings[i]) for i in range(len(db.names))
            ]

            sim_vecs.append(np.max(sims))
            labels.append(1)

        # Unknown probes
        for img_path in unknown_imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            emb = None
            if model_wrapper.name == "arcface":
                faces = model_wrapper.app.get(img)
                if faces:
                    emb = faces[0].embedding.astype(np.float32)
            else:
                faces = model_wrapper.detector.detect(img)
                if faces:
                    aligned = model_wrapper.detector.align_for(img, faces[0]["kps"])
                    if aligned is not None:
                        emb = model_wrapper.embed(aligned)

            if emb is None:
                continue

            sims = [
                cosine_similarity(emb, db.embeddings[i]) for i in range(len(db.names))
            ]

            sim_vecs.append(np.max(sims))
            labels.append(0)

    sim_vecs = np.array(sim_vecs, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # Thresholds to test
    thresholds = np.linspace(0, 1, 400)
    tpir_at_fpir = compute_open_set_metrics(sim_vecs, labels, thresholds)

    return {
        "tpir_at_fpir": tpir_at_fpir,
    }


# ------------------------- Runtime -------------------------


def measure_runtime(model_wrapper, db, dataset_root):
    persons = list_identities(dataset_root)

    det_times = []
    align_times = []
    emb_times = []
    match_times = []
    total_times = []

    for person_dir in persons:
        img_files = [
            p
            for p in person_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not img_files:
            continue

        for img_path in img_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            t0 = time.time()

            # detect
            if model_wrapper.name == "arcface":
                td0 = time.time()
                faces = model_wrapper.app.get(img)
                td1 = time.time()
                det_times.append(td1 - td0)

                if not faces:
                    continue

                # No alignment needed
                aligned = img
                align_times.append(0.0)

                # embed
                te0 = time.time()
                emb = faces[0].embedding.astype(np.float32)
                te1 = time.time()
                emb_times.append(te1 - te0)

            else:
                # detect
                td0 = time.time()
                faces = model_wrapper.detector.detect(img)
                td1 = time.time()
                det_times.append(td1 - td0)

                if not faces:
                    continue

                # align
                ta0 = time.time()
                aligned = model_wrapper.detector.align_for(img, faces[0]["kps"])
                ta1 = time.time()
                align_times.append(ta1 - ta0)

                if aligned is None:
                    continue

                # embed
                te0 = time.time()
                emb = model_wrapper.embed(aligned)
                te1 = time.time()
                emb_times.append(te1 - te0)

            if emb is None:
                continue

            # match
            tm0 = time.time()
            sims = [
                cosine_similarity(emb, db.embeddings[i]) for i in range(len(db.names))
            ]
            _ = np.max(sims)
            tm1 = time.time()

            match_times.append(tm1 - tm0)

            total_times.append(time.time() - t0)

    return {
        "det_mean_ms": float(np.mean(det_times) * 1000) if det_times else 0,
        "align_mean_ms": float(np.mean(align_times) * 1000) if align_times else 0,
        "emb_mean_ms": float(np.mean(emb_times) * 1000) if emb_times else 0,
        "match_mean_ms": float(np.mean(match_times) * 1000) if match_times else 0,
        "total_mean_ms": float(np.mean(total_times) * 1000) if total_times else 0,
        "throughput_fps": float(1.0 / np.mean(total_times)) if total_times else 0,
    }


# ------------------------- Main -------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="arcface / adaface_camera / facenet_camera"
    )
    parser.add_argument("--dataset", required=True, help="Path to CUSTOM_DATASET")
    parser.add_argument("--db", required=False, help="Optional path to db_<model>.npz")
    args = parser.parse_args()

    model_wrapper = load_model(args.model)

    # Load DB
    if args.db:
        db_path = args.db
    else:
        db_path = Path(__file__).resolve().parent / f"db_{args.model}.npz"

    print(f"[ID] Loading DB: {db_path}")
    db = FaceEmbeddingDB.load(db_path)

    # Ensure export directory
    exports_dir = Path(__file__).resolve().parent / "exports_id"
    exports_dir.mkdir(exist_ok=True)

    print("\n[ID] Running CLOSED-SET evaluation...")
    closed = evaluate_closed_set(model_wrapper, db, args.dataset)

    print("\n[ID] Running OPEN-SET evaluation...")
    open_set = evaluate_open_set(model_wrapper, db, args.dataset)

    print("\n[ID] Measuring RUNTIME...")
    runtime = measure_runtime(model_wrapper, db, args.dataset)

    # Combine results
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "closed_set": closed,
        "open_set": open_set,
        "runtime": runtime,
    }

    # Save JSON
    json_path = exports_dir / f"{args.model}_identification_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[ID] Saved JSON: {json_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
