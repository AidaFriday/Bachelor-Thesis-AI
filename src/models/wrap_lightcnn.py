import os
import sys
import torch
import numpy as np
import torch.nn.functional as F

# --- add LightCNN to path ---
LIGHTCNN_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../external/LightCNN")
)
sys.path.append(LIGHTCNN_PATH)

from light_cnn_v4 import LightCNN_29Layers_v2  # use most accurate version


class LightCNNWrapper:
    """Wrapper for pretrained LightCNN (v4 / 29v2)."""

    def __init__(self, model_variant="LightCNN-29v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LightCNN_29Layers_v2(num_classes=80013)
        weights_path = os.path.join(LIGHTCNN_PATH, "models", f"{model_variant}.pth")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"❌ Missing pretrained weights: {weights_path}")

        state = torch.load(weights_path, map_location=self.device)
        if "state_dict" in state:
            state = state["state_dict"]

        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, batch: np.ndarray) -> np.ndarray:
        """Compute normalized LightCNN embeddings from input batch (N,H,W,C)."""
        # Convert to grayscale if RGB
        if batch.ndim == 4 and batch.shape[-1] == 3:
            batch = np.mean(batch, axis=-1, keepdims=True)

        x = (
            torch.from_numpy(batch.transpose(0, 3, 1, 2))
            .float()
            .div(255.0)
            .to(self.device)
        )
        feats, _ = self.model(x)
        feats = F.normalize(feats)
        return feats.cpu().numpy()
