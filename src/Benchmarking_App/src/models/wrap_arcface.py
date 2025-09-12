import os
from insightface.app import FaceAnalysis


class ArcFaceWrapper:
    name = "arcface"

    def __init__(self, device: str, model_path: str, input_size=(112, 112)):
        ctx_id = 0 if device == "cuda" else -1
        self.app = FaceAnalysis(root=os.path.dirname(model_path), name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.input_size = input_size

    def detect_and_embed(self, frame):
        """Return faces with bbox, kps, embedding."""
        faces = self.app.get(frame)
        results = []
        for f in faces:
            results.append({
                "bbox": f.bbox.astype(int),
                "kps": f.kps.astype(float),
                "embedding": f.embedding
            })
        return results
