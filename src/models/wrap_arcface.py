import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis


class ArcFaceWrapper:
    """
    ArcFace wrapper using insightface buffalo_l pack.
    Reference: Deng et al., ArcFace: Additive Angular Margin Loss, CVPR 2019.
    """

    name = "arcface"

    def __init__(
        self,
        device: str = "cpu",
        model_path: str = "src/models/pretrained_models",
        input_size=(112, 112),
    ):
        self.device = device
        self.input_size = input_size

        # If user points directly to buffalo_l, strip it to get the parent
        if model_path.endswith("buffalo_l"):
            root_path = os.path.dirname(model_path)
            buffalo_dir = model_path
        else:
            root_path = model_path
            buffalo_dir = os.path.join(model_path, "buffalo_l")

        # Ensure the buffalo_l folder exists (insightface will download here if missing)
        os.makedirs(buffalo_dir, exist_ok=True)

        ctx_id = 0 if device == "cuda" else -1

        print(f"[ArcFace] Using buffalo_l at: {buffalo_dir}")

        # InsightFace FaceAnalysis provides detection + embedding
        self.detector = FaceAnalysis(root=root_path, name="buffalo_l")
        self.detector.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def detect_and_embed(self, frame: np.ndarray):
        """Detect faces and return bbox, landmarks, and ArcFace embeddings."""
        faces = self.detector.get(frame)
        results = []
        for f in faces:
            results.append(
                {
                    "bbox": f.bbox.astype(int),
                    "kps": f.kps.astype(float),
                    "embedding": f.embedding.astype(np.float32),
                }
            )
        return results

    def get_embedding(self, img_path: str) -> np.ndarray:
        """
        Load image, detect main face, and return ArcFace embedding.
        Falls back to center crop + no alignment if detector fails.
        """
        frame = cv2.imread(img_path)
        if frame is None:
            raise ValueError(f"[ArcFace] Could not read image: {img_path}")

        faces = self.detect_and_embed(frame)
        if len(faces) > 0:
            return faces[0]["embedding"]
