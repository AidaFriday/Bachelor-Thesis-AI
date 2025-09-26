import os, sys
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis


def import_magface(model_path):
    """Import MagFace backbone from local repo."""
    if not model_path or not os.path.isdir(model_path):
        raise RuntimeError(f"[MagFace] Repo not found at: {model_path}")

    sys.path.insert(0, model_path)
    sys.path.insert(0, os.path.join(model_path, "models"))

    try:
        from iresnet import iresnet100  # Backbone used in MagFace
    except ImportError as e:
        raise RuntimeError(
            f"[MagFace] Could not import iresnet backbone. Check repo structure. Original error: {e}"
        )

    return iresnet100


class MagFaceWrapper:
    """
    MagFace wrapper with IResNet backbone.
    Reference: MagFace: A Universal Representation for Face Recognition and Quality Assessment (CVPR 2021).
    """

    name = "magface"

    def __init__(self, device: str = "cpu", model_path: str = None, input_size=(112, 112)):
        self.device = torch.device(device)
        self.input_size = input_size

        # Resolve project root (the "src" directory)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "pretrained_models", "magface")

        # Import backbone
        IResNet100 = import_magface(model_path)

        # Initialize and load pretrained checkpoint (if available)
        self.model = IResNet100().to(self.device).eval()

        ckpt_path = os.path.join(model_path, "magface_epoch_00025.pth")
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in sd:
                sd = {
                    k.replace("module.", "").replace("backbone.", ""): v
                    for k, v in sd["state_dict"].items()
                }
            self.model.load_state_dict(sd, strict=False)
        else:
            print(f"[MagFace] Warning: no checkpoint found at {ckpt_path}, using random init.")

        # Face detector
        ctx_id = 0 if self.device.type == "cuda" else -1
        self.detector = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
        self.detector.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        """Return normalized MagFace embedding for a cropped face (BGR)."""
        rgb = cv2.cvtColor(cv2.resize(bgr, self.input_size), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        t = (t - 0.5) / 0.5
        with torch.inference_mode():
            feat = (
                self.model(t.unsqueeze(0).to(self.device))[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        feat /= np.linalg.norm(feat) + 1e-12
        return feat

    def detect_and_embed(self, frame: np.ndarray):
        """Detect faces in a frame and return bbox, landmarks, and embeddings."""
        faces = self.detector.get(frame)
        results = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            results.append(
                {
                    "bbox": f.bbox.astype(int),
                    "kps": f.kps.astype(float),
                    "embedding": self.embed(crop),
                }
            )
        return results

    def get_embedding(self, img_path: str) -> np.ndarray:
        """
        Load image, detect main face, and return MagFace embedding.
        Falls back to center crop if detector fails.
        """
        frame = cv2.imread(img_path)
        if frame is None:
            raise ValueError(f"[MagFace] Could not read image: {img_path}")

        faces = self.detect_and_embed(frame)
        if len(faces) > 0:
            return faces[0]["embedding"]

        # fallback: use center crop
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        crop = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]
        return self.embed(crop)
