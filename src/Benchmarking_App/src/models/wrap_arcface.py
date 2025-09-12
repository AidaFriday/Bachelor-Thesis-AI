import os
from insightface.app import FaceAnalysis


class ArcFaceWrapper:
    """
    ArcFace wrapper using insightface buffalo_l pack.
    Pretrained weights auto-download into ~/.insightface if not found.
    Reference: Deng et al., ArcFace: Additive Angular Margin Loss, CVPR 2019.
    """
    name = "arcface"

    def __init__(self, device: str, model_path: str, input_size=(112, 112)):
        self.device = device
        self.input_size = input_size

        if not os.path.isdir(model_path):
            raise RuntimeError(f"[ArcFace] Model path not found: {model_path}")

        ctx_id = 0 if device == "cuda" else -1
        self.detector = FaceAnalysis(root=os.path.dirname(model_path), name="buffalo_l")
        self.detector.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def detect_and_embed(self, frame):
        """Detect faces and return bbox, landmarks, and ArcFace embeddings."""
        faces = self.detector.get(frame)
        results = []
        for f in faces:
            results.append({
                "bbox": f.bbox.astype(int),
                "kps": f.kps.astype(float),
                "embedding": f.embedding.astype(float)
            })
        return results
