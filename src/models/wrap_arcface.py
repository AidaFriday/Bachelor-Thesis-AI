import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis


class ArcFaceWrapper:
    """
    ArcFace wrapper using insightface buffalo_l models.
    """

    name = "arcface"

    def __init__(self, device: str = "cpu", input_size=(112, 112)):
        self.device = device
        self.input_size = tuple(input_size)

        ctx_id = 0 if device == "cuda" else -1

        # keep same object but expose it as both app and detector
        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        # 👈 this makes old code that uses wrapper.detector still work
        self.detector = self.app

        # optional env toggle from your older code (no harm keeping it)
        self._force_embed_only = os.getenv("FORCE_EMBED_ONLY", "0") == "1"

        # grab recognition model
        self._rec = None
        models = getattr(self.app, "models", None)
        if isinstance(models, dict):
            self._rec = models.get("recognition", None)

        # determine recognition input size (fallback to 112x112)
        self._rec_input = self.input_size
        if self._rec is not None:
            ishape = getattr(self._rec, "input_size", None)
            if isinstance(ishape, (tuple, list)) and len(ishape) == 2:
                self._rec_input = (int(ishape[0]), int(ishape[1]))

    # ---------- NEW: embedding for LFW-style aligned faces ----------
    def embed_aligned(self, bgr: np.ndarray) -> np.ndarray:
        """
        Embedding for an already cropped + aligned face (e.g. LFW-deepfunneled).
        This avoids running detection.
        """
        if bgr is None:
            return None

        if self._rec is None:
            # fall back to generic embed (which may use detection)
            return self.embed(bgr)

        W, H = self._rec_input
        face = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        if hasattr(self._rec, "get"):
            emb = self._rec.get(face)
        elif hasattr(self._rec, "get_feat"):
            emb = self._rec.get_feat(face)
        elif hasattr(self._rec, "forward"):
            emb = self._rec.forward(face)
        else:
            emb = None

        if emb is None:
            return None

        return np.asarray(emb, dtype=np.float32).reshape(-1)

    # ------- detection + embedding (for camera / misc) ----------
    def detect_and_embed(self, frame: np.ndarray):
        faces = self.app.get(frame)
        results = []
        for f in faces:
            emb = getattr(f, "embedding", None)
            if emb is None:
                continue
            results.append(
                {
                    "bbox": f.bbox.astype(int),
                    "kps": f.kps.astype(float),
                    "embedding": emb.astype(np.float32),
                }
            )
        return results

    # ------- generic embed (kept for backwards compatibility) ----------
    def embed(self, bgr: np.ndarray) -> np.ndarray:
        """
        Embedding for a face crop. Uses the recognition backbone when possible,
        otherwise falls back to detection.
        """
        if bgr is None:
            return None

        # try direct recognition first
        if self._rec is not None:
            try:
                W, H = self._rec_input
                face = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                if hasattr(self._rec, "get"):
                    emb = self._rec.get(face)
                elif hasattr(self._rec, "get_feat"):
                    emb = self._rec.get_feat(face)
                elif hasattr(self._rec, "forward"):
                    emb = self._rec.forward(face)
                else:
                    emb = None

                if emb is not None:
                    return np.asarray(emb, dtype=np.float32).reshape(-1)
            except Exception:
                # fall through to detector
                pass

        # fallback: run detector and use first face embedding
        faces = self.app.get(bgr)
        if len(faces) > 0 and getattr(faces[0], "embedding", None) is not None:
            return faces[0].embedding.astype(np.float32)
        return None

    # ------- convenience for path-based tests ----------
    def get_embedding(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            return None

        # for generic tests we keep old behaviour: detect + embed
        faces = self.detect_and_embed(img)
        if faces:
            return faces[0]["embedding"]

        # fallback: treat whole image as a face crop
        return self.embed(img)
