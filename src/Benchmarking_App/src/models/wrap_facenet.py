import os, sys
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis


class FaceNetWrapper:
    """
    FaceNet wrapper (InceptionResnetV1).
    Loads model from local facenet repo in pretrained_models/facenet.
    Reference: Schroff et al., FaceNet (CVPR 2015).
    """
    name = "facenet"

    def __init__(self, device: str, model_path: str, input_size=(160, 160)):
        self.device = torch.device(device)
        self.input_size = input_size

        if not model_path or not os.path.isdir(model_path):
            raise RuntimeError(f"[FaceNet] Local repo not found at: {model_path}")

        # Add parent of facenet to sys.path → so we can `import facenet.models...`
        facenet_parent = os.path.dirname(model_path)  # .../pretrained_models
        if facenet_parent not in sys.path:
            sys.path.insert(0, facenet_parent)

        try:
            from facenet.models.inception_resnet_v1 import InceptionResnetV1
        except ImportError as e:
            raise RuntimeError(f"[FaceNet] Could not import inception_resnet_v1: {e}")

        # Load pretrained weights (downloads if not cached)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # RetinaFace detector from insightface
        ctx_id = 0 if self.device.type == "cuda" else -1
        self.detector = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
        self.detector.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        """Return FaceNet embedding for a cropped face (BGR)."""
        rgb = cv2.cvtColor(cv2.resize(bgr, self.input_size), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        t = (t - 0.5) / 0.5
        with torch.inference_mode():
            emb = self.model(t.unsqueeze(0).to(self.device))[0].detach().cpu().numpy().astype(np.float32)
        return emb

    def detect_and_embed(self, frame):
        """Detect faces and return bbox, landmarks, and embeddings."""
        faces = self.detector.get(frame)
        results = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            results.append({
                "bbox": f.bbox.astype(int),
                "kps": f.kps.astype(float),
                "embedding": self.embed(crop)
            })
        return results
