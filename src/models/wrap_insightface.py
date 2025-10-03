import cv2
import numpy as np
import insightface


class InsightFaceWrapper:
    """
    Wrapper around InsightFace (ArcFace, CosFace, etc).
    Handles detection + embedding.
    """

    name = "insightface"

    def __init__(self, device: str = "cpu", det_size=(640, 640), input_size=(112, 112)):
        self.device = device
        self.input_size = input_size  # ✅ added so performance.py & accuracy.py can use it

        # FaceAnalysis auto-downloads pretrained models to ~/.insightface
        ctx_id = 0 if device == "cuda" else -1
        self.model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
        self.model.prepare(ctx_id=ctx_id, det_size=det_size)
        print(f"[InsightFaceWrapper] Loaded InsightFace with det_size={det_size}")

    def detect_and_embed(self, frame: np.ndarray):
        """
        Detect faces in an image and return list of dicts with bbox + embedding.
        """
        faces = self.model.get(frame)
        results = []
        for f in faces:
            results.append(
                {
                    "bbox": np.array(f.bbox, dtype=np.int32),
                    "kps": f.kps if hasattr(f, "kps") else np.zeros((5, 2)),
                    "embedding": np.array(f.embedding, dtype=np.float32),
                }
            )
        return results

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        """
        Get embedding of a cropped face (no detection).
        """
        faces = self.model.get(bgr)
        if len(faces) > 0:
            return np.array(faces[0].embedding, dtype=np.float32)
        return None

    def get_embedding(self, img_path: str) -> np.ndarray:
        """
        Load image from path and return first face embedding.
        """
        frame = cv2.imread(img_path)
        if frame is None:
            raise ValueError(f"[InsightFaceWrapper] Cannot read image: {img_path}")
        faces = self.detect_and_embed(frame)
        if faces:
            return faces[0]["embedding"]
        return None
