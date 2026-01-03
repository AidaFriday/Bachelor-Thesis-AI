# models/wrap_adaface.py

import os, sys

# ✅ Make external/adaface_repo importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "external"))

import cv2
import torch
import numpy as np

from models.wrap_facedetection import FaceDetectorAligner
from adaface_repo.net import IR_50


class AdaFaceOriginalWrapper:
    name = "adaface_original"

    def __init__(self, device: str = "cpu", input_size=(112, 112)):
        """
        device: "cpu" or "cuda:0" etc.
        input_size: backbone input resolution (H, W).
        """
        self.device = device
        self.input_size = tuple(map(int, input_size))

        ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        model_path = os.path.join(
            ROOT, "external", "adaface_repo", "adaface_ir50_ms1mv2", "model.pt"
        )

        # ✅ Backbone
        self.model = IR_50(self.input_size)

        ckpt = torch.load(model_path, map_location=device)

        # Some checkpoints store weights under "net"
        if "net" in ckpt:
            ckpt = ckpt["net"]

        # Remove "net." prefix if exists in keys
        new_state = {}
        for k, v in ckpt.items():
            new_key = k.replace("net.", "")
            new_state[new_key] = v

        self.model.load_state_dict(new_state, strict=False)
        self.model.to(device)
        self.model.eval()

        # ✅ Shared detector/aligner (used by the framework & ROC/LFW script)
        self.detector = FaceDetectorAligner(device=device)

    # ---------- core: embedding for an aligned face crop ----------
    def _forward_aligned(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Internal helper: assumes img_bgr is a *face crop* (already roughly aligned).
        Resizes to self.input_size, preprocesses, runs backbone, L2-normalizes.
        """
        if img_bgr is None:
            return None

        # Resize to backbone input size
        img = cv2.resize(img_bgr, self.input_size, interpolation=cv2.INTER_AREA)

        # BGR -> RGB, float32, AdaFace normalization
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)[None, :]  # (1, C, H, W)
        img = torch.from_numpy(img).float().to(self.device)

        # Forward
        with torch.no_grad():
            emb = self.model(img)[0]

        # L2-normalize
        emb = emb / emb.norm(p=2)

        return emb.cpu().numpy().astype(np.float32)

    # ---------- LFW-style: already aligned crop ----------
    def embed_aligned(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Embedding for an *already cropped + aligned* face (e.g. LFW-deepfunneled).
        No detection is performed here.
        """
        return self._forward_aligned(img_bgr)

    # ---------- framework: embed(aligned_crop) ----------
    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Generic embedding method expected by the rest of your framework.
        Assumes the input is already a face crop (often aligned by FaceDetectorAligner).
        """
        return self._forward_aligned(img_bgr)

    # ---------- convenience: detect + align + embed ----------
    def detect_and_embed(self, frame: np.ndarray):
        """
        Detect faces in a full frame, align them with FaceDetectorAligner,
        then embed each face.

        Useful for camera / non-benchmark scripts.
        """
        faces = self.detector.detect(frame)
        results = []

        for f in faces:
            kps = f.get("kps", None)
            if kps is None:
                continue

            aligned = self.detector.align_for(frame, kps)
            if aligned is None:
                continue

            emb = self.embed(aligned)
            if emb is None:
                continue

            results.append(
                {
                    "bbox": f["bbox"],
                    "kps": kps,
                    "embedding": emb,
                }
            )

        return results
