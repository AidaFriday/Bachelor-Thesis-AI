import numpy as np
import cv2
from insightface.app import FaceAnalysis


class ArcFaceWrapper:
    """
    ArcFace wrapper using insightface buffalo_l package models.
    """

    name = "arcface"

    def __init__(self, device: str = "cpu", input_size=(112, 112)):
        self.device = device
        self.input_size = input_size

        ctx_id = 0 if device == "cuda" else -1
        self.detector = FaceAnalysis(name="buffalo_l")
        self.detector.prepare(ctx_id=ctx_id, det_size=(640, 640))

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

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        """
        Get embedding of a cropped face (no detection).
        """
        faces = self.detector.get(bgr)
        if len(faces) > 0 and getattr(faces[0], "embedding", None) is not None:
            return faces[0].embedding.astype(np.float32)
        return None

    def get_embedding(self, img_path: str) -> np.ndarray:
        frame = cv2.imread(img_path)
        if frame is None:
            raise ValueError(f"[ArcFace] Could not read image: {img_path}")

        faces = self.detect_and_embed(frame)
        if faces:
            return faces[0]["embedding"]

        # fallback: center crop then embed
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        crop = frame[(h - min_dim)//2:(h + min_dim)//2,
                     (w - min_dim)//2:(w + min_dim)//2]
        return self.embed(crop)
