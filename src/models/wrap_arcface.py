import numpy as np
import cv2
import os  # ADDED: for env toggles
from insightface.app import FaceAnalysis


class ArcFaceWrapper:
    """
    ArcFace wrapper using insightface buffalo_l package models.
    """

    name = "arcface"

    def __init__(self, device: str = "cpu", input_size=(112, 112)):
        self.device = device
        self.input_size = tuple(input_size)

        # Keep FaceAnalysis for detection paths (camera, etc.)
        ctx_id = 0 if device == "cuda" else -1
        self.detector = FaceAnalysis(  # CHANGED: allow recognition too
            name="buffalo_l", allowed_modules=["detection", "recognition"]  # ADDED
        )
        self.detector.prepare(ctx_id=ctx_id, det_size=(640, 640))

        # ADDED: optional env toggles (do not break existing code)
        # FORCE_EMBED_ONLY=1 -> embed() only (no detection). detect_and_embed() unchanged.
        # FORCE_DETECT=1     -> (not used here; kept for future symmetry)
        self._force_embed_only = os.getenv("FORCE_EMBED_ONLY", "0") == "1"

        # ADDED: try to get recognition model to embed aligned crops directly
        self._rec = None
        try:
            models = getattr(self.detector, "models", None)
            if isinstance(models, dict):
                self._rec = models.get("recognition", None)
            if self._rec is None and hasattr(self.detector, "model_zoo"):
                for m in getattr(self.detector, "model_zoo", []):
                    if getattr(m, "taskname", "") == "recognition":
                        self._rec = m
                        break
        except Exception:
            self._rec = None

        # ADDED: determine recognition input size (fallback to 112x112)
        self._rec_input = self.input_size
        try:
            ishape = getattr(self._rec, "input_size", None)
            if isinstance(ishape, (tuple, list)) and len(ishape) == 2:
                self._rec_input = (int(ishape[0]), int(ishape[1]))
        except Exception:
            pass

    # --------- unchanged public API for camera etc. ----------
    def detect_and_embed(self, frame: np.ndarray):
        faces = self.detector.get(frame)
        results = []
        for f in faces:
            if getattr(f, "embedding", None) is None:
                continue
            results.append(
                {
                    "bbox": f.bbox.astype(int),
                    "kps": f.kps.astype(float),
                    "embedding": f.embedding.astype(np.float32),
                }
            )
        return results

    # ---------- embedding-only on already aligned/cropped face ----------
    def embed(self, bgr: np.ndarray) -> np.ndarray:
        """
        Return embedding from an ALREADY aligned + cropped face.
        No detection is run here (fast & low-variance).
        """
        # ADDED: try direct recognition model call
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

                if emb is None:
                    return None
                return np.asarray(emb, dtype=np.float32).reshape(-1)
            except Exception:
                # fall back below
                pass

        # CHANGED: final fallback — use detector only if direct recognition path failed
        faces = self.detector.get(bgr)
        if len(faces) > 0 and getattr(faces[0], "embedding", None) is not None:
            return faces[0].embedding.astype(np.float32)
        return None

    def get_embedding(self, img_path: str) -> np.ndarray:
        frame = cv2.imread(img_path)
        if frame is None:
            raise ValueError(f"[ArcFace] Could not read image: {img_path}")

        # keep existing behavior (detection) for convenience callers
        faces = self.detect_and_embed(frame)
        if faces:
            return faces[0]["embedding"]

        # fallback: center crop then embed (assumes the crop is a face)
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        crop = frame[
            (h - min_dim) // 2 : (h + min_dim) // 2,
            (w - min_dim) // 2 : (w + min_dim) // 2,
        ]
        return self.embed(crop)
