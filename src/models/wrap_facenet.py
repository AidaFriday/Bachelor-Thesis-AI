import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN


class FaceNetWrapper:
    """
    FaceNet wrapper using facenet-pytorch package.
    """

    name = "facenet"

    def __init__(self, device: str = "cpu", input_size=(160, 160)):
        self.device = torch.device(device)
        self.input_size = input_size

        # Load pretrained InceptionResnetV1
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.detector = MTCNN(image_size=self.input_size[0], device=self.device)

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(bgr, self.input_size), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        t = (t - 0.5) / 0.5
        with torch.inference_mode():
            emb = self.model(t.unsqueeze(0).to(self.device))
        return emb[0].cpu().numpy().astype(np.float32)

    def detect_and_embed(self, frame: np.ndarray):
        boxes, probs = self.detector.detect(frame)
        results = []
        if boxes is not None:
            for box in boxes.astype(int):
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                results.append(
                    {
                        "bbox": box,
                        "kps": np.zeros((5, 2)),  # MTCNN landmarks optional
                        "embedding": self.embed(crop),
                    }
                )
        return results

    def get_embedding(self, img_path: str) -> np.ndarray:
        frame = cv2.imread(img_path)
        if frame is None:
            raise ValueError(f"[FaceNet] Could not read image: {img_path}")
        faces = self.detect_and_embed(frame)
        if faces:
            return faces[0]["embedding"]

        # fallback: center crop
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        crop = frame[(h - min_dim)//2:(h + min_dim)//2,
                     (w - min_dim)//2:(w + min_dim)//2]
        return self.embed(crop)
