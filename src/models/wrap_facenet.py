# old good facenet wrapper

# models/wrap_facenet.py

import os
import torch
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1
from typing import List, Dict

from .wrap_facedetection import FaceDetectorAligner


class FaceNetWrapper:
    name = "facenet"

    def __init__(self, device=None, input_size=(160, 160)):
        # device: "cpu" / "cuda"
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.input_size = tuple(map(int, input_size))
        self._embed_wh = (self.input_size[0], self.input_size[1])

        # ✅ Pretrained FaceNet (VGGFace2)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # ✅ Universal detector / aligner
        self.aligner = FaceDetectorAligner(device=self.device.type)

        # ✅ For framework compatibility (like AdaFace / ArcFace)
        self.detector = self.aligner

    # ---------- internal: forward on an aligned face crop ----------
    @torch.no_grad()
    def _forward_aligned(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Assumes img_bgr is already a face crop (aligned or roughly centered).
        Resizes, normalizes, runs FaceNet, and L2-normalizes the embedding.
        """
        if img_bgr is None:
            return None

        # Resize and convert to RGB
        img = cv2.resize(img_bgr, self._embed_wh, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0

        # (H,W,C) -> (1,C,H,W)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        emb = self.model(tensor)
        emb = emb / emb.norm(dim=1, keepdim=True)

        return emb.cpu().numpy().flatten().astype(np.float32)

    # ---------- for LFW / aligned datasets ----------
    def embed_aligned(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Embedding for *already cropped + aligned* faces (e.g. LFW-deepfunneled).
        No detection performed here.
        """
        return self._forward_aligned(img_bgr)

    # ---------- generic embed (framework expects this) ----------
    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Embedding for a face crop (usually aligned by FaceDetectorAligner).
        """
        return self._forward_aligned(img_bgr)

    # ---------- detect → align → embed (for raw images, camera, etc.) ----------
    def detect_and_embed(self, frame: np.ndarray):
        """
        Detect faces in a full frame, align them, and return embeddings.
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
            if emb is None:
                continue

            results.append(
                {
                    "bbox": f["bbox"],
                    "embedding": emb,
                }
            )
        return results
