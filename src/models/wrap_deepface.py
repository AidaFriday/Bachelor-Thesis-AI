import cv2
import numpy as np
import torch
from typing import List, Dict

# ✅ Import the ArcFace module (not a class)
from deepface.models.facial_recognition import ArcFace

from .wrap_facedetection import FaceDetectorAligner


class DeepFaceWrapper:
    name = "deepface"

    def __init__(self, device=None, input_size=(112, 112)):
        # Auto-select GPU if available
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = tuple(input_size)

        # ✅ Load ArcFace model correctly
        self.model = ArcFace.load_model()

        # ✅ Create aligner
        self.aligner = FaceDetectorAligner(device=self.device)

    # ----- Single face embedding -----
    def embed(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")

        # ✅ ArcFace normalization standard
        img = (img - 127.5) / 128.0

        img = np.expand_dims(img, axis=0)
        emb = self.model.predict(img)[0]
        emb = emb / np.linalg.norm(emb)
        return emb.astype(np.float32)

    # ----- Detect → Align → Embed -----
    def detect_and_embed(self, frame: np.ndarray) -> List[Dict]:
        faces = self.aligner.detect(frame)
        results = []

        for face in faces:
            crop = self.aligner.align_for(frame, face["kps"])
            if crop is None:
                continue

            emb = self.embed(crop)

            results.append({"bbox": face["bbox"], "kps": face["kps"], "embedding": emb})

        return results
