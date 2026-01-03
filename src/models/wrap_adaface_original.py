@ -0,0 +1,72 @@
import os, sys

# ✅ Make external/adaface_repo importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "external"))

import cv2
import torch
import numpy as np

from models.wrap_facedetection import FaceDetectorAligner
from adaface_repo.net import IR_50


class AdaFaceWrapper:
    name = "adaface"

    def __init__(self, device="cpu", input_size=(112, 112)):
        self.device = device
        self.input_size = tuple(input_size)

        ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        model_path = os.path.join(
            ROOT, "external", "adaface_repo", "adaface_ir50_ms1mv2", "model.pt"
        )

        # ✅ Correct backbone load
        # ✅ Correct backbone load
        self.model = IR_50(self.input_size)

        ckpt = torch.load(model_path, map_location=device)

        # Some checkpoints store weights under "net"
        if "net" in ckpt:
            ckpt = ckpt["net"]

        # Remove "net." prefix if exists in keys
        new_state = {}
        for k, v in ckpt.items():
            new_key = k.replace("net.", "")  # strip prefix
            new_state[new_key] = v

        self.model.load_state_dict(new_state, strict=False)
        self.model.to(device)
        self.model.eval()

        # Will be overwritten by connector (shared), but safe fallback
        self.detector = FaceDetectorAligner(device=device)

    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        # ---------- Detect & Align (system-compatible) ----------
        faces = self.detector.get(img_bgr, out_size=self.input_size)

        if len(faces) == 0:
            img = cv2.resize(img_bgr, self.input_size)
        else:
            img = faces[0]  # Already aligned

        # ---------- Preprocess ----------
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)[None, :]
        img = torch.from_numpy(img).float().to(self.device)

        # ---------- Forward ----------
        with torch.no_grad():
            emb = self.model(img)[0]

        # ---------- Normalize ----------
        emb = emb / emb.norm(p=2)

        return emb.cpu().numpy().astype(np.float32)