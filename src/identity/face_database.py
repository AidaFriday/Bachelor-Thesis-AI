import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from connector import load_model


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class FaceEmbeddingDB:
    """
    Simple embedding database:
      - names: list[str]
      - embeddings: (N, D) float32
      - threshold: similarity threshold for 'Unknown'
    """

    def __init__(
        self, names: List[str], embeddings: np.ndarray, threshold: float = 0.35
    ):
        self.names = list(names)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.threshold = float(threshold)

    # --------------- persistence -----------------
    @classmethod
    def load(cls, path: os.PathLike, threshold: float = 0.35) -> "FaceEmbeddingDB":
        data = np.load(path, allow_pickle=True)
        names = list(data["names"])
        embeddings = data["embeddings"].astype(np.float32)
        return cls(names, embeddings, threshold=threshold)

    def save(self, path: os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, names=np.array(self.names, dtype=object), embeddings=self.embeddings
        )

    # --------------- recognition -----------------
    def match(self, emb: np.ndarray) -> Tuple[str, float]:
        """
        Return (name, similarity). If similarity < threshold -> name == 'Unknown'.
        """
        if self.embeddings.size == 0:
            return "Unknown", 0.0

        q = np.asarray(emb, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return "Unknown", 0.0

        db_norms = np.linalg.norm(self.embeddings, axis=1) + 1e-8
        sims = (self.embeddings @ q) / (db_norms * q_norm)
        idx = int(np.argmax(sims))
        best_sim = float(sims[idx])

        if best_sim < self.threshold:
            return "Unknown", best_sim
        return self.names[idx], best_sim


# ============================================================
# Builder: from CUSTOM_DATASET style folder to .npz database
# ============================================================


def build_database_from_folder(
    dataset_root: str,
    model_name: str,
    out_path: str,
    max_images_per_person: int | None = 10,
) -> FaceEmbeddingDB:
    """
    dataset_root structure:
        dataset_root/
            Alice/ img1.jpg, img2.png, ...
            Bob/   ...
    """
    dataset_root = Path(dataset_root)
    out_path = Path(out_path)

    wrapper = load_model(model_name)
    print(f"[DB] Using model '{wrapper.name}' to build database from: {dataset_root}")

    all_names: List[str] = []
    all_means: List[np.ndarray] = []

    people = [d for d in sorted(dataset_root.iterdir()) if d.is_dir()]
    if not people:
        raise RuntimeError(f"No person folders found in {dataset_root}")

    for person_dir in people:
        person_name = person_dir.name
        img_paths = [
            p
            for p in sorted(person_dir.iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not img_paths:
            print(f"[DB] WARNING: no images for {person_name}, skipping.")
            continue

        print(f"[DB] Person '{person_name}': {len(img_paths)} images")

        embs = []
        for i, img_path in enumerate(img_paths):
            if max_images_per_person is not None and i >= max_images_per_person:
                break

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[DB]   ! Could not read {img_path}, skipping")
                continue

            faces = wrapper.detector.detect(img)
            if not faces:
                print(f"[DB]   ! No face found in {img_path}, skipping")
                continue

            kps = faces[0]["kps"]
            aligned = wrapper.detector.align_for(img, kps)
            if aligned is None:
                print(f"[DB]   ! Alignment failed for {img_path}, skipping")
                continue

            emb = wrapper.embed(aligned)
            embs.append(emb)

        if not embs:
            print(f"[DB] WARNING: no valid embeddings for {person_name}, skipping.")
            continue

        mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
        all_names.append(person_name)
        all_means.append(mean_emb)
        print(f"[DB]   -> kept {len(embs)} embeddings for {person_name}")

    if not all_names:
        raise RuntimeError("No valid person embeddings were built; database is empty.")

    embeddings = np.stack(all_means, axis=0).astype(np.float32)
    db = FaceEmbeddingDB(all_names, embeddings)
    db.save(out_path)

    print(f"[DB] Saved database with {len(all_names)} identities to: {out_path}")
    return db


if __name__ == "__main__":
    # CLI helper: build DB from terminal
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Model name (e.g. arcface, facenet, adaface)"
    )
    parser.add_argument("--dataset", required=True, help="Path to CUSTOM_DATASET root")
    parser.add_argument(
        "--out",
        required=False,
        default=None,
        help="Output .npz path (default: src/identity/db_<model>.npz)",
    )
    parser.add_argument(
        "--max_per_person",
        type=int,
        default=10,
        help="Max images per person to use",
    )
    args = parser.parse_args()

    src_root = Path(__file__).resolve().parents[1]
    default_out = src_root / "identity" / f"db_{args.model}.npz"
    out_path = args.out or default_out

    build_database_from_folder(
        args.dataset, args.model, out_path, max_images_per_person=args.max_per_person
    )
