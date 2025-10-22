import os, sys, torch, numpy as np, torch.nn.functional as F

SPHEREFACE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../external/sphereface_pytorch")
)
sys.path.append(SPHEREFACE_PATH)

from net_sphere import sphere20a


class SphereFaceWrapper:
    """Interface identical to ArcFaceWrapper/FacenetWrapper"""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = sphere20a()
        weights = os.path.join(SPHEREFACE_PATH, "model", "sphere20a_20171020.pth")
        ckpt = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(ckpt)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def infer(self, batch: np.ndarray) -> np.ndarray:
        """Input (N,H,W,C) RGB → output (N,512) L2-normalized embeddings"""
        x = (
            torch.from_numpy(batch.transpose(0, 3, 1, 2))
            .float()
            .div(255.0)
            .to(self.device)
        )
        feats = F.normalize(self.model(x))
        return feats.cpu().numpy()
