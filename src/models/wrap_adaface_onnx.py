import os
import cv2
import numpy as np
import onnxruntime as ort
import torch

from models.wrap_facedetection import FaceDetectorAligner


class AdaFaceONNX:
    def __init__(self, onnx_path="models/adaface.onnx", device="cuda"):
        # Select device
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        # ONNX providers
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )

        print(f"[ADA-ONNX] Loading {onnx_path}")
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # IO names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Shared InsightFace detector
        self.detector = FaceDetectorAligner(device=self.device)

        print(f"[ADA-ONNX] Providers → {self.session.get_providers()}")

    # ---------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------
    def _preprocess(self, img_bgr):
        img = cv2.resize(img_bgr, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = img.transpose(2, 0, 1)  # HWC → CHW
        img = np.expand_dims(img, axis=0)
        return img

    # ---------------------------------------------------------------
    # Inference on aligned image
    # ---------------------------------------------------------------
    def embed(self, img_bgr):
        inp = self._preprocess(img_bgr)
        out = self.session.run([self.output_name], {self.input_name: inp})[0]
        return out[0]

    # ---------------------------------------------------------------
    # Full pipeline: detect → align → embed
    # ---------------------------------------------------------------
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
