import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from connector import load_model


# ============================================================
# Cosine similarity helper
# ============================================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ============================================================
# Face Embedding Database
# ============================================================
class FaceEmbeddingDB:
    """
    Embedding DB:
      - names: list[str]
      - embeddings: shape (N, D)
      - match() returns best name + similarity
    """

    def __init__(
        self, names: List[str], embeddings: np.ndarray, threshold: float = 0.35
    ):
        # names list
        self.names = list(names)

        # embeddings -> force proper shape
        emb = np.asarray(embeddings, dtype=np.float32)

        # FIX SHAPE (N,1,D) → (N,D)
        if emb.ndim == 3 and emb.shape[1] == 1:
            print("[DB] Fixing embeddings shape (N,1,D) → (N,D)")
            emb = emb[:, 0, :]

        # FIX SHAPE (N,D,1) → (N,D)
        if emb.ndim == 3 and emb.shape[2] == 1:
            print("[DB] Fixing embeddings shape (N,D,1) → (N,D)")
            emb = emb[:, :, 0]

        self.embeddings = emb.astype(np.float32)
        self.threshold = float(threshold)

    # ---------------- persistence ----------------
    @classmethod
    def load(cls, path: os.PathLike, threshold: float = 0.35) -> "FaceEmbeddingDB":
        data = np.load(path, allow_pickle=True)
        names = list(data["names"])
        emb = data["embeddings"].astype(np.float32)
        return cls(names, emb, threshold=threshold)

    def save(self, path: os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            names=np.array(self.names, dtype=object),
            embeddings=self.embeddings.astype(np.float32),
        )

    # ---------------- recognition ----------------
    def match(self, emb, threshold=None):
        """
        emb: (D,) or (1,D) or (D,1)
        Returns: (best_name, best_similarity)
        """
        try:
            # normalize emb shape
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)

            # DB shape already cleaned in __init__
            db = self.embeddings

            # compute cosine similarity to all rows
            sims = np.dot(db, emb).reshape(-1)

            # best index
            idx = int(np.argmax(sims))
            best_sim = float(sims[idx])
            best_name = self.names[idx]

            # apply threshold
            th = threshold if threshold is not None else self.threshold
            if best_sim < th:
                return "Unknown", best_sim

            return best_name, best_sim

        except Exception as e:
            print(f"[match ERROR] {e}")
            return "Unknown", 0.0


# ============================================================
# Database Builder
# ============================================================
def build_database_from_folder(
    dataset_root: str,
    model_name: str,
    out_path: str,
    max_images_per_person: int | None = 10,
) -> FaceEmbeddingDB:

    dataset_root = Path(dataset_root)
    wrapper = load_model(model_name)

    print(f"[DB] Using model '{wrapper.name}' to build DB from: {dataset_root}")

    all_names = []
    all_means = []

    # People folders
    people = [d for d in sorted(dataset_root.iterdir()) if d.is_dir()]
    if not people:
        raise RuntimeError(f"No person folders found at {dataset_root}")

    for person_dir in people:
        person_name = person_dir.name
        img_paths = sorted(
            [
                p
                for p in person_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
            ]
        )

        if not img_paths:
            print(f"[DB] No images for {person_name} — skipping")
            continue

        print(f"[DB] Person '{person_name}': {len(img_paths)} images")

        embs = []

        for i, img_path in enumerate(img_paths):
            if max_images_per_person is not None and i >= max_images_per_person:
                break

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[DB] ! Cannot read {img_path}")
                continue

            # 1) detect
            faces = wrapper.detector.detect(img)
            if not faces:
                print(f"[DB] ! No face in {img_path}")
                continue

            # 2) align
            kps = faces[0]["kps"]
            aligned = wrapper.detector.align_for(img, kps)
            if aligned is None:
                print(f"[DB] ! Align fail {img_path}")
                continue

            # 3) embed
            emb = wrapper.embed(aligned)

            # FIX: ALWAYS flatten to (512,)
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)

            embs.append(emb)

        if not embs:
            print(f"[DB] WARNING: no valid embeddings for {person_name}")
            continue

        mean_emb = np.mean(np.stack(embs, axis=0), axis=0)

        all_names.append(person_name)
        all_means.append(mean_emb)

        print(f"[DB]   -> kept {len(embs)} embeddings for {person_name}")

    if not all_names:
        raise RuntimeError("DB is empty — no valid embeddings.")

    embeddings = np.stack(all_means, axis=0).astype(np.float32)
    db = FaceEmbeddingDB(all_names, embeddings)
    db.save(out_path)

    print(f"[DB] Saved DB with {len(all_names)} identities → {out_path}")
    return db


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max_per_person", type=int, default=10)

    args = parser.parse_args()

    src_root = Path(__file__).resolve().parents[1]
    default_out = src_root / "identity" / f"db_{args.model}.npz"
    out_path = args.out or default_out

    build_database_from_folder(
        args.dataset,
        args.model,
        out_path,
        max_images_per_person=args.max_per_person,
    )
