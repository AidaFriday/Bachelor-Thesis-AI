import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


class FaceNetONNX:
    """
    FaceNet Inception-ResNet-V1 (128-dim) ONNX wrapper.
    Supports CPU + CUDA + TensorRT.
    """

    def __init__(self, device="cpu"):

        # ----------------------------------------------------------
        # Resolve ONNX model path
        # src/models → src/ → project root → external/FaceNet_onnx/
        # ----------------------------------------------------------
        ROOT = Path(__file__).resolve().parents[2]     # project root
        model_path = ROOT / "external" / "FaceNet_onnx" / "facenet_irv1_128.onnx"

        if not model_path.exists():
            raise FileNotFoundError(f"[FaceNet-ONNX] Model not found: {model_path}")

        self.model_path = str(model_path)

        # ----------------------------------------------------------
        # Choose GPU or CPU providers
        # ----------------------------------------------------------
        if device.lower() == "cuda":
            providers = [
                "CUDAExecutionProvider",
                "TensorrtExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        print(f"[FaceNetONNX] Loading: {self.model_path}")
        print(f"[FaceNetONNX] Providers: {providers}")

        # ----------------------------------------------------------
        # Create ONNX Runtime session
        # ----------------------------------------------------------
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception as e:
            print("\n[WARNING] GPU initialization failed → Falling back to CPU")
            print("Reason:", e)
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    # ----------------------------------------------------------
    # PREPROCESSING
    # ----------------------------------------------------------
    def preprocess(self, img_bgr):
        """
        Convert BGR face crop to normalized 160×160 NCHW float32 tensor.
        """
        img = cv2.resize(img_bgr, (160, 160))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5       # normalize to [-1, 1]

        img = img.transpose(2, 0, 1)  # CHW
        img = np.expand_dims(img, 0)  # NCHW
        return img

    # ----------------------------------------------------------
    # EMBEDDING
    # ----------------------------------------------------------
    def embed(self, img_bgr):
        """
        Forward pass + L2 normalization.
        Input: aligned face BGR
        Output: 128-dim float32 embedding
        """
        inp = self.preprocess(img_bgr)
        out = self.session.run([self.output_name], {self.input_name: inp})[0]
        emb = out[0]

        # L2 normalize
        emb = emb / np.linalg.norm(emb)
        return emb.astype(np.float32)

    # ----------------------------------------------------------
    # COSINE SIMILARITY
    # ----------------------------------------------------------
    @staticmethod
    def cosine_similarity(e1, e2):
        """
        Cosine similarity in [-1, 1].
        """
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
