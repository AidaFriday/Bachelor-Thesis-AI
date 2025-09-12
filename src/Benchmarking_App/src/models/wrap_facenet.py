import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from insightface.app import FaceAnalysis


class FaceNetWrapper:
    name = "facenet"

    def __init__(self, device: str, model_path: str = None, input_size=(160, 160)):
        self.device = torch.device(device)
        self.input_size = input_size
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        ctx_id = 0 if device == "cuda" else -1
        self.detector = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
        self.detector.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(bgr, self.input_size), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        t = (t - 0.5) / 0.5
        with torch.inference_mode():
            v = self.model(t.unsqueeze(0).to(self.device))[0].detach().cpu().numpy().astype(np.float32)
        return v

    def detect_and_embed(self, frame):
        faces = self.detector.get(frame)
        results = []
        for f in faces:
            results.append({
                "bbox": f.bbox.astype(int),
                "kps": f.kps.astype(float),
                "embedding": self.embed(frame)
            })
        return results
