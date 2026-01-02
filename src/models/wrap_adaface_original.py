import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from wrap_facedetection import FaceDetectorAligner


class AdaFaceOriginalWrapper:
    name = "adaface_original"

    def __init__(self, device="cuda", input_size=(112, 112), model_path=None):
        self.input_size = tuple(input_size)

        # -------------------------------------------------
        # Resolve AdaFace repo path
        # -------------------------------------------------
        if model_path is None:
            model_path = (
                Path(__file__).resolve().parents[2]
                / "external"
                / "adaface_repo"
            )
        model_path = str(Path(model_path).resolve())

        self.device = self._pick_device(device)

        # -------------------------------------------------
        # Load AdaFace IR-50
        # -------------------------------------------------
        sys.path.insert(0, model_path)
        try:
            from net import IR_50
        finally:
            sys.path.pop(0)

        self.model = IR_50(input_size=self.input_size)

        ckpt_path = os.path.join(
            model_path, "adaface_ir50_ms1mv2", "model.pt"
        )
        state_dict = torch.load(ckpt_path, map_location="cpu")

        if any(k.startswith("net.") for k in state_dict):
            state_dict = {
                k.replace("net.", "", 1): v
                for k, v in state_dict.items()
            }

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # -------------------------------------------------
        # IMPORTANT: detector + aligner (REQUIRED FOR ~99%)
        # -------------------------------------------------
        self.detector = FaceDetectorAligner(device=str(self.device))

    # -------------------------------------------------
    def _pick_device(self, requested: str) -> torch.device:
        req = (requested or "cpu").lower()
        if req.startswith("cuda") and torch.cuda.is_available():
            try:
                archs = torch.cuda.get_arch_list()
            except Exception:
                archs = []

            if "sm_120" not in archs:
                print(
                    "[AdaFaceOriginal] WARNING: GPU sm_120 unsupported, "
                    "falling back to CPU."
                )
                return torch.device("cpu")

            return torch.device("cuda")

        return torch.device("cpu")

    # -------------------------------------------------
    def preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        img = cv2.resize(img_bgr, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).unsqueeze(0).to(self.device)

    # -------------------------------------------------
    @torch.no_grad()
    def embed_aligned(self, img_bgr: np.ndarray) -> np.ndarray:
        x = self.preprocess(img_bgr)

        # AdaFace forward returns (feat, norm)
        feat = self.model(x)[0]
        feat = torch.nn.functional.normalize(feat, dim=1)

        emb = feat.cpu().numpy().reshape(-1).astype(np.float32)

        if emb.shape[0] != 512:
            raise RuntimeError("AdaFace produced invalid embedding")

        return emb

    # -------------------------------------------------
    @torch.no_grad()
    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        OFFICIAL AdaFace evaluation path:
        image → detect → align → embed
        """

        faces = self.detector.detect(img_bgr)
        if not faces:
            raise RuntimeError("No face detected")

        aligned = self.detector.align_for(img_bgr, faces[0]["kps"])
        if aligned is None:
            raise RuntimeError("Alignment failed")

        return self.embed_aligned(aligned)
