import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as T
import torch


class FaceNetOriginalWrapper:
    """
    Wrapper for the original FaceNet (InceptionResnetV1 pretrained on vggface2).
    Matches the embedding logic used inside your LFW protocol.
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.model = InceptionResnetV1(pretrained="vggface2").eval()
        self.name = "facenet_original"
        self.detector = None  # no built-in detector; dataset must be aligned

        # Preprocessing pipeline (matches your LFW logic)
        self.transform = T.Compose(
            [T.ToTensor(), T.Resize((160, 160)), T.Normalize([0.5], [0.5])]
        )

    def embed(self, img: np.ndarray) -> np.ndarray:
        """
        Accepts BGR numpy image (OpenCV), converts to RGB, applies preprocessing,
        and returns the 512-d FaceNet embedding.
        """
        if img is None:
            raise ValueError("Input image is None")

        # Convert BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t = self.transform(img_rgb).unsqueeze(0)
        with torch.no_grad():
            emb = self.model(t).numpy().flatten()

        return emb


# Loader interface used by connector.load_model()
def load():
    return FaceNetOriginalWrapper()
