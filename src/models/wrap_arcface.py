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

    def __init__(self, device: str = "cpu", model_path: str = None, input_size=(112, 112)):
        self.device = device
        self.input_size = input_size

        # Resolve project root (the "src" directory)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(
                base_dir, "models", "pretrained_models", "arcface_repo", "models"
            )
        model_path = os.path.abspath(model_path)

        # Expected buffalo_l directory
        buffalo_dir = os.path.join(model_path, "buffalo_l")
        if not os.path.isdir(buffalo_dir):
            raise RuntimeError(
                f"[ArcFace] buffalo_l not found at {buffalo_dir}. "
                "Run download_arcface_models.py to fetch it."
            )

        ctx_id = 0 if device == "cuda" else -1
        print(f"[ArcFace] Using buffalo_l at: {buffalo_dir}")

        # InsightFace FaceAnalysis provides detection + embedding
        self.detector = FaceAnalysis(root=model_path, name="buffalo_l")
        self.detector.prepare(ctx_id=ctx_id, det_size=(640, 640))

        if (
            "recognition" not in self.detector.models
            or self.detector.models["recognition"] is None
        ):
            raise RuntimeError(
                "[ArcFace] Recognition model not loaded. "
                f"Check {buffalo_dir}/w600k_r50.onnx"
            )

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

    def get_embedding(self, img_path: str) -> np.ndarray:
        frame = cv2.imread(img_path)
        if frame is None:
            raise ValueError(f"[ArcFace] Could not read image: {img_path}")

        faces = self.detect_and_embed(frame)
        if faces:
            return faces[0]["embedding"]

        # fallback: center crop
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        crop = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]
        return self.detector.models["recognition"].get(crop).astype(np.float32)
