# ==== models/wrap_sphereface.py ====
import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Optional

# Use the same detector as your FaceNet wrapper for identical behavior
from facenet_pytorch import MTCNN

# ---- SphereFace (sphere20a) ----
import sys

SPHEREFACE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../external/sphereface_pytorch")
)
if SPHEREFACE_PATH not in sys.path:
    sys.path.append(SPHEREFACE_PATH)
from net_sphere import sphere20a


class SphereFaceWrapper:
    """
    SphereFace wrapper (sphere20a) with the exact app-facing behavior as FaceNetWrapper:
      - name = "sphereface"
      - embed(bgr) -> np.ndarray : embedding-only on an already cropped/aligned face
      - detect_and_embed(frame) -> List[Dict] : MTCNN detect → crop → embed
      - get_embedding(img_path) -> np.ndarray : convenience, mirrors FaceNetWrapper
    """

    name = "sphereface"

    def __init__(
        self,
        device: str = "cpu",
        input_size=(112, 112),
        weights_path: Optional[str] = None,
    ):
        # match FaceNetWrapper's device style
        self.device = torch.device(device)
        self.input_size = tuple(input_size)

        # --- Load SphereFace model ---
        # allow explicit path via arg or environment variable
        explicit = weights_path or os.getenv("SPHEREFACE_WEIGHTS")

        # common defaults
        candidates = []
        if explicit:
            candidates.append(explicit)
        candidates += [
            os.path.join(SPHEREFACE_PATH, "model", "sphere20a_20171020.pth"),
            os.path.join(SPHEREFACE_PATH, "model", "sphere20a.pth"),
        ]

        ckpt_path = next((p for p in candidates if p and os.path.exists(p)), None)
        if ckpt_path is None:
            msg = (
                "❌ Missing SphereFace pretrained weights.\n"
                "Tried:\n  - " + "\n  - ".join(candidates) + "\n\n"
                "Fix one of these:\n"
                "  • Place weights at one of the paths above, OR\n"
                "  • Set environment variable SPHEREFACE_WEIGHTS to your .pth file.\n"
            )
            raise FileNotFoundError(msg)

        self.model = sphere20a().to(self.device).eval()
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(
            state
        )  # sphere20a checkpoints are usually a plain state dict

        # --- Detector (identical to FaceNetWrapper’s approach) ---
        # Use MTCNN with the same image size convention
        self.detector = MTCNN(image_size=self.input_size[0], device=self.device)

    # ---------- helpers ----------
    def _to_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        """
        Resize to self.input_size, convert BGR→RGB, and to torch tensor in [0,1].
        Keep preprocessing intentionally similar to FaceNetWrapper to match behavior.
        """
        rgb = cv2.cvtColor(cv2.resize(bgr, self.input_size), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # C,H,W in [0,1]
        return t

    # ---------- public API ----------
    def embed(self, bgr: np.ndarray) -> np.ndarray:
        """
        Embedding-only on an already aligned/cropped face (no detection).
        Mirrors FaceNetWrapper: resize → convert → forward.
        Output is L2-normalized float32 vector.
        """
        t = self._to_tensor(bgr).unsqueeze(0).to(self.device)  # 1,C,H,W
        with torch.inference_mode():
            feat = self.model(t)  # (1, D)
            feat = F.normalize(feat, dim=1)  # L2 normalize for cosine sims
        return feat[0].cpu().numpy().astype(np.float32)

    def detect_and_embed(self, frame: np.ndarray) -> List[Dict]:
        """
        Detection path for camera/production (same structure as FaceNetWrapper):
          - MTCNN detect
          - crop bbox
          - embed on the crop
          - return dicts with bbox, dummy kps (MTCNN kps optional), embedding
        """
        boxes, probs = self.detector.detect(frame)
        results: List[Dict] = []
        if boxes is not None:
            for box in boxes.astype(int):
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                results.append(
                    {
                        "bbox": box,
                        "kps": np.zeros(
                            (5, 2)
                        ),  # keep identical shape as FaceNetWrapper
                        "embedding": self.embed(crop),
                    }
                )
        return results

    def get_embedding(self, img_path: str) -> np.ndarray:
        """
        Mirrors FaceNetWrapper:
          - read image
          - run detect_and_embed
          - fallback to center crop → embed if no detection
        """
        frame = cv2.imread(img_path)
        if frame is None:
            raise ValueError(f"[SphereFace] Could not read image: {img_path}")
        faces = self.detect_and_embed(frame)
        if faces:
            return faces[0]["embedding"]

        # fallback: center crop (same as your FaceNet wrapper)
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        y0 = (h - min_dim) // 2
        x0 = (w - min_dim) // 2
        crop = frame[y0 : y0 + min_dim, x0 : x0 + min_dim]
        return self.embed(crop)
