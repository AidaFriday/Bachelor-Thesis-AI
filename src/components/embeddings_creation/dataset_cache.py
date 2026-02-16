import numpy as np
from pathlib import Path


class DatasetEmbeddingCache:
    """
    Loads precomputed embeddings (.npz) and performs cosine similarity matching.
    Must use the SAME model that created the database.
    """

    def __init__(self, wrapper, dataset_path=None):
        self.wrapper = wrapper
        self.dataset_path = dataset_path

        self.embeddings = None
        self.labels = None
        self.paths = None

        self.names = []

        # --------------------------------------------------
        # Database location (THIS FOLDER)
        # --------------------------------------------------
        self.base_dir = Path(__file__).parent

        self.model_to_file = {
            "arcface": self.base_dir / "faces_arcface.npz",
            "facenet_camera": self.base_dir / "faces_facenet.npz",
            "adaface_camera": self.base_dir / "faces_adaface.npz",
        }

    # --------------------------------------------------
    def load_or_build(self):
        model_name = self.wrapper.name

        if model_name not in self.model_to_file:
            raise RuntimeError(
                f"No embedding database for model '{model_name}'"
            )

        db_path = self.model_to_file[model_name]

        if not db_path.exists():
            raise RuntimeError(
                f"Embedding file not found: {db_path}"
            )

        data = np.load(db_path)

        self.embeddings = data["embeddings"].astype(np.float32)
        self.labels = data["labels"]
        self.paths = data["paths"]

        # Normalize (safety)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings /= norms

        self.names = sorted(set(self.labels.tolist()))

        print(
            f"[DB] Loaded {len(self.embeddings)} embeddings "
            f"({len(self.names)} identities) from {db_path.name}"
        )

    # --------------------------------------------------
    def match(self, emb: np.ndarray):
        """
        Returns best match (name, cosine_similarity)
        """
        if self.embeddings is None:
            return "Unknown", 0.0

        emb = emb.astype(np.float32)
        emb /= np.linalg.norm(emb)

        sims = np.dot(self.embeddings, emb)
        idx = int(np.argmax(sims))

        return self.labels[idx], float(sims[idx])
