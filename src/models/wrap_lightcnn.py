# models/wrap_lightcnn.py
import os
import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Dict

# ---- Ensure LightCNN source is always visible ----
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../external/LightCNN"))
)

# ---- External LightCNN path ----
LIGHTCNN_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../external/LightCNN")
)
if LIGHTCNN_PATH not in sys.path:
    sys.path.append(LIGHTCNN_PATH)

# ✅ This import now always works (even when launched from main.py)
from light_cnn_v4 import LightCNN_V4 as LightCNN_29Layers_v2


# ---- Detector for alignment (InsightFace RetinaFace) ----
from insightface.app import FaceAnalysis


def _align_by_5pts(bgr: np.ndarray, kps: np.ndarray, out_size=(128, 128)) -> np.ndarray:
    """
    Simple 5-point similarity alignment to a square canvas,
    keeps it robust and consistent across models.
    """
    # reference from ArcFace (scaled to target size)
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
    ref = ref * scale
    M = cv2.estimateAffinePartial2D(kps.astype(np.float32), ref, method=cv2.LMEDS)[0]
    if M is None:
        return None
    return cv2.warpAffine(bgr, M, out_size, flags=cv2.INTER_LINEAR)


class LightCNNWrapper:
    """
    Wrapper for pretrained LightCNN (29v2) with a unified API:
      - name
      - embed(bgr) -> np.ndarray
      - detect_and_embed(frame) -> List[Dict]
    """

    name = "lightcnn"

    def __init__(
        self, device=None, model_variant="LightCNN-29v2", input_size=(128, 128)
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = tuple(input_size)

        # --- model ---
        self.model = LightCNN_29Layers_v2(None)

        weights_path = os.path.join(LIGHTCNN_PATH, "models", f"{model_variant}.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"❌ Missing pretrained weights: {weights_path}")

        state = torch.load(weights_path, map_location=self.device)
        if "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()

        # --- face detector (for raw images) ---
        ctx_id = 0 if str(self.device).startswith("cuda") else -1
        self.det = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
        self.det.prepare(ctx_id=ctx_id, det_size=(640, 640))

    @torch.no_grad()
    def _infer_batch(self, batch: np.ndarray) -> np.ndarray:
        if batch.ndim != 4:
            raise ValueError("Expected batch with shape (N,H,W,C)")

        # make sure we feed 3 channels (LightCNN v4 expects 3-channel BGR / 255)
        if batch.shape[-1] == 1:
            batch = np.repeat(batch, 3, axis=-1)

        x = (
            torch.from_numpy(batch.transpose(0, 3, 1, 2))
            .float()
            .div(255.0)
            .to(self.device)
        )

        out = self.model(x)  # v4 may return just features
        feats = out[0] if isinstance(out, (tuple, list)) else out
        feats = F.normalize(feats)  # L2-normalize
        return feats.cpu().numpy()

    # ---------- public API ----------
    def embed(self, bgr: np.ndarray) -> np.ndarray:
        """
        Accepts either an aligned face crop or a full image.
        If it looks unaligned, we run detection+alignment first.
        """
        H, W = bgr.shape[:2]
        # Heuristic: if much bigger than target, try to detect & align
        if max(H, W) > max(self.input_size) * 1.3:
            faces = self.det.get(bgr)
            if faces:
                f = faces[0]
                kps = np.array(f.kps, dtype=np.float32)
                crop = _align_by_5pts(bgr, kps, out_size=self.input_size)
            else:
                # fallback: center crop square
                s = min(H, W)
                y0 = (H - s) // 2
                x0 = (W - s) // 2
                crop = bgr[y0 : y0 + s, x0 : x0 + s]
                crop = cv2.resize(crop, self.input_size, interpolation=cv2.INTER_AREA)
        else:
            crop = cv2.resize(bgr, self.input_size, interpolation=cv2.INTER_AREA)

        crop = crop.astype(np.uint8)
        batch = crop[None, ...]  # (1,H,W,C)
        return self._infer_batch(batch)[0].astype(np.float32)

    def detect_and_embed(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces, align, and return a list with bbox, kps, embedding.
        """
        faces = self.det.get(frame)
        results = []
        for f in faces:
            kps = np.array(f.kps, dtype=np.float32)
            crop = _align_by_5pts(frame, kps, out_size=self.input_size)
            if crop is None:
                continue
            emb = self.embed(crop)  # uses fast path (already aligned)
            results.append(
                {
                    "bbox": np.array(f.bbox, dtype=np.int32),
                    "kps": kps,
                    "embedding": emb.astype(np.float32),
                }
            )
        return results
