# wrap_facenet_onnx.py
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


class FaceNetONNX:
    """
    FaceNet Inception-ResNet-V1 (128-dim) ONNX wrapper.
    GPU + CPU. No TensorRT.
    """

    def __init__(self, device="cpu"):

        ROOT = Path(__file__).resolve().parents[2]
        model_path = ROOT / "external" / "FaceNet_onnx" / "facenet_irv1_128.onnx"

        if not model_path.exists():
            raise FileNotFoundError(f"[FaceNet-ONNX] Model not found: {model_path}")

        self.model_path = str(model_path)

        # GPU OR CPU — NO TENSORRT
        if device.lower() == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        print(f"[FaceNetONNX] Loading: {self.model_path}")
        print(f"[FaceNetONNX] Providers: {providers}")

        self.session = ort.InferenceSession(self.model_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    # ----------------------------------------------------------
    def preprocess(self, img_bgr):
        img = cv2.resize(img_bgr, (160, 160))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        return img

    # ----------------------------------------------------------
    def embed(self, img_bgr):
        inp = self.preprocess(img_bgr)
        out = self.session.run([self.output_name], {self.input_name: inp})[0]
        emb = out[0]
        emb = emb / np.linalg.norm(emb)
        return emb.astype(np.float32)

    # ----------------------------------------------------------
    @staticmethod
    def cosine_similarity(e1, e2):
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
