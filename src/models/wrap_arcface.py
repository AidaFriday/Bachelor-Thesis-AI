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
        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        # grab recognition model
        self._rec = None
        models = getattr(self.app, "models", None)
        if isinstance(models, dict):
            self._rec = models.get("recognition", None)

        # default input size
        self._rec_input = self.input_size
        if self._rec is not None:
            ishape = getattr(self._rec, "input_size", None)
            if isinstance(ishape, (tuple, list)) and len(ishape) == 2:
                self._rec_input = (int(ishape[0]), int(ishape[1]))

    # -------- detection + embedding (for camera / misc) ----------
    def detect_and_embed(self, frame: np.ndarray):
        faces = self.app.get(frame)
        results = []
        for f in faces:
            emb = getattr(f, "embedding", None)
            if emb is None:
                continue
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
            n = np.linalg.norm(emb)
            if n > 0:
                emb /= n
            results.append(
                {
                    "bbox": f.bbox.astype(int),
                    "kps": f.kps.astype(float),
                    "embedding": emb,
                }
            )
        return results

    # -------- embedding of an already aligned face crop ----------
    def embed(self, bgr: np.ndarray) -> np.ndarray:
        """
        img: aligned face crop, BGR uint8
        returns: L2-normalized embedding (float32) or None
        """
        if bgr is None or self._rec is None:
            return None

        W, H = self._rec_input
        # NOTE: cv2.resize expects (width, height)
        face = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)

        # IMPORTANT: ArcFaceONNX.get expects BGR, so no cvtColor here
        emb = self._rec.get(face)
        if emb is None:
            return None

        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(emb)
        if n > 0:
            emb /= n
        return emb

    def get_embedding(self, img_path: str) -> np.ndarray:
        frame = cv2.imread(img_path)
        if frame is None:
            raise ValueError(f"[ArcFace] Could not read image: {img_path}")

        faces = self.detect_and_embed(frame)
        if faces:
            return faces[0]["embedding"]

        # fallback: center crop then direct embed
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        crop = frame[
            (h - min_dim) // 2 : (h + min_dim) // 2,
            (w - min_dim) // 2 : (w + min_dim) // 2,
        ]
        return self.embed(crop)
