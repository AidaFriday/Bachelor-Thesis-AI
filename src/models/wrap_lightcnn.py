# models/wrap_lightcnn.py
import os
import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Dict

# ---- Ensure LightCNN source path ----
SYS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../external/LightCNN")
)
if SYS_ROOT not in sys.path:
    sys.path.append(SYS_ROOT)

# ---- Correct architecture for LightCNN-29v2 ----
from light_cnn import LightCNN_29Layers_v2

# ---- Detector for alignment ----
from insightface.app import FaceAnalysis


def _align_by_5pts(bgr: np.ndarray, kps: np.ndarray, out_size=(128, 128)) -> np.ndarray:
    """5-point similarity alignment."""
    ref = np.float32(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ]
    )
    scale = out_size[0] / 112.0
    ref *= scale
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ref, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(bgr, M, out_size, flags=cv2.INTER_LINEAR)


class LightCNNWrapper:
    """
    Wrapper for pretrained LightCNN-29v2.
    """

    name = "lightcnn"

    def __init__(
        self, device=None, model_variant="LightCNN-29v2", input_size=(128, 128)
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.input_size = tuple(map(int, input_size))
        self._embed_wh = (self.input_size[0], self.input_size[1])

        # --- Model init ---
        self.model = LightCNN_29Layers_v2()  # correct architecture

        weights_path = os.path.join(SYS_ROOT, "models", f"{model_variant}.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"❌ Expected weights file not found:\n{weights_path}"
            )

        state = torch.load(weights_path, map_location=self.device)
        if "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)

        self.model.to(self.device).eval()

        # --- Face detector for raw frames ---
        ctx_id = 0 if self.device.type == "cuda" else -1
        self.det = FaceAnalysis(
            name="buffalo_l", allowed_modules=["detection", "landmark"]
        )
        self.det.prepare(ctx_id=ctx_id, det_size=(640, 640))

    # ---------------- PUBLIC API ----------------

    def embed(self, inp: np.ndarray) -> np.ndarray:
        """
        Accepts:
          (H,W,3) BGR aligned
        or (1,1,128,128) LightCNN-format tensor
        """

        # Case: already preprocessed (1,1,128,128)
        if inp.ndim == 4 and inp.shape[1:] == (1, 128, 128):
            tensor = torch.from_numpy(inp).float().to(self.device)

        # Case: raw aligned BGR image (H,W,3)
        elif inp.ndim == 3 and inp.shape[2] == 3:
            crop = cv2.resize(inp, self._embed_wh, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = gray.astype("float32") / 255.0
            tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device)

        else:
            raise ValueError(f"❌ embed() received invalid shape: {inp.shape}")

        with torch.no_grad():
            out = self.model(tensor)
            feats = out[0] if isinstance(out, (tuple, list)) else out
            feats = F.normalize(feats, dim=1)

        return feats.cpu().numpy().flatten().astype(np.float32)

    def detect_and_embed(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect → align → embed from a raw frame.
        """
        faces = self.det.get(frame)
        results = []

        for f in faces:
            kps = np.array(f.kps, dtype=np.float32)
            crop = _align_by_5pts(frame, kps, out_size=self._embed_wh)
            if crop is None:
                continue

            emb = self.embed(crop)
            results.append(
                {
                    "bbox": np.array(f.bbox, dtype=np.int32),
                    "kps": kps,
                    "embedding": emb,
                }
            )
        return results
