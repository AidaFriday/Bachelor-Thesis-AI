# models/wrap_facenet_camera.py

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from facenet_pytorch import InceptionResnetV1

# Reuse original Facenet logic (safe, untouched)
from models.wrap_facenet_original import FaceNetOriginalWrapper

# Reuse existing YOLOv5 face detection + alignment
from models.wrap_facedetection import FaceDetectorAligner


class FaceNetCameraWrapper(FaceNetOriginalWrapper):
    """
    Facenet + YOLOv5 detection/alignment for CAMERA and DATABASE building.
    This does NOT modify wrap_facenet_original.py.
    """

    def __init__(self, device="cpu"):
        super().__init__(device)

        # Add detector (original wrapper has none)
        self.detector = FaceDetectorAligner(device)

        # Facenet expects 160×160 aligned faces
        self.output_size = (160, 160)

        # Distinguish this model from facenet_original
        self.name = "facenet_camera"

    def detect_and_embed(self, frame):
        faces = self.detector.detect(frame)
        results = []

        for f in faces:
            bbox, kps = f["bbox"], f["kps"]
            aligned = self.detector.align_for(frame, kps, self.output_size)
            if aligned is None:
                continue

            emb = self.embed(aligned)

            results.append(
                {
                    "bbox": bbox,
                    "kps": kps,
                    "embedding": emb,
                }
            )

        return results
