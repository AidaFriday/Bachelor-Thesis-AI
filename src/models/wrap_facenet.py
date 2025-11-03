# models/wrap_facenet.py

import torch
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1
from typing import List, Dict

from .wrap_facedetection import FaceDetectorAligner


class FaceNetWrapper:
    name = "facenet"

    def __init__(self, device=None, input_size=(160, 160)):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.input_size = tuple(map(int, input_size))
        self._embed_wh = (self.input_size[0], self.input_size[1])

        # ✅ Load pretrained FaceNet weights (L2-normalized embedding output)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # ✅ Use your universal aligner
        self.aligner = FaceDetectorAligner(device=self.device.type)

    @torch.no_grad()
    def embed(self, img):
        """
        img: must be (H, W, 3) BGR aligned face
        returns 512-D embedding (float32, normalized)
        """
        if img is None:
            raise ValueError("embed() called with None image")

        # Resize and convert to RGB
        img = cv2.resize(img, self._embed_wh, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0

        # (H,W,C) -> (1,C,H,W)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        emb = self.model(tensor)
        emb = emb / emb.norm(dim=1, keepdim=True)

        return emb.cpu().numpy().flatten().astype(np.float32)

    def detect_and_embed(self, frame):
        """
        Detect faces → align using detected keypoints → embed
        """
        faces = self.aligner.detect(frame)
        results: List[Dict] = []
        for f in faces:
            kps = f.get("kps", None)
            if kps is None:
                continue

            crop = self.aligner.align_for(frame, kps)
            if crop is None:
                continue

            emb = self.embed(crop)
            results.append(
                {
                    "bbox": f["bbox"],
                    "embedding": emb,
                }
            )
        return results
