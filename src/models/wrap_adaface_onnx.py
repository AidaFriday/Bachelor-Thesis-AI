import os
import cv2
import numpy as np
import onnxruntime as ort
import torch
from pathlib import Path

from models.wrap_facedetection import FaceDetectorAligner


class AdaFaceONNX:
    def __init__(self, device="cuda"):
        # ------------------------------------------------------------------
        # Resolve ONNX model path RELATIVE to repo root
        # ------------------------------------------------------------------
        ROOT = Path(__file__).resolve().parents[2]   # → Bachelor-Thesis-AI/
        onnx_path = ROOT / "external" / "adaface_onnx" / "adaface.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(f"[ERROR] ONNX model not found: {onnx_path}")

        onnx_path = str(onnx_path)

        # ------------------------------------------------------------------
        # Select device
        # ------------------------------------------------------------------
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )

        print(f"[ADA-ONNX] Loading {onnx_path}")

        # ------------------------------------------------------------------
        # Load ONNX model
        # ------------------------------------------------------------------
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Shared InsightFace detector
        self.detector = FaceDetectorAligner(device=self.device)

        print(f"[ADA-ONNX] Providers → {self.session.get_providers()}")

    # ----------------------------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------------------------
    def _preprocess(self, img_bgr):
        img = cv2.resize(img_bgr, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        # CORRECT AdaFace normalization
        img = img / 255.0
        img = (img - 0.5) / 0.5

        img = img.transpose(2, 0, 1)[None, ...]  # (1,3,112,112)
        return img


    # ----------------------------------------------------------------------
    # Inference on aligned image
    # ----------------------------------------------------------------------
    def embed(self, img_bgr):
        inp = self._preprocess(img_bgr)
        out = self.session.run([self.output_name], {self.input_name: inp})[0]

        # out can be (1, D) or (D,) depending on the model/export
        emb = out[0] if out.ndim == 2 else out.squeeze()

        # L2-normalize (VERY important for AdaFace/verification)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        #print("[ADA-ONNX] emb norm =", float(np.linalg.norm(emb)))
        return emb.astype(np.float32)

        



    # ----------------------------------------------------------------------
    # Full pipeline: detect → align → embed
    # ----------------------------------------------------------------------
    def get_embedding(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None

        faces = self.detector.detect(img)
        if not faces:
            return None

        aligned = self.detector.align_for(img, faces[0]["kps"])
        if aligned is None:
            return None

        return self.embed(aligned)
