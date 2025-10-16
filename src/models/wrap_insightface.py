import cv2
import numpy as np
import onnxruntime as ort
import insightface
import os  # ADDED: env toggles


class InsightFaceWrapper:
    """
    Wrapper around InsightFace (ArcFace, CosFace, etc).
    Handles detection + embedding. embed() now assumes aligned crop.
    """

    name = "insightface"

    def __init__(self, device: str = "cpu", det_size=(640, 640), input_size=(112, 112)):
        self.device = device
        self.input_size = tuple(input_size)

        # Pick provider automatically (unchanged)
        available = ort.get_available_providers()
        if "TensorrtExecutionProvider" in available:
            providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            print("[InsightFaceWrapper] Using TensorRT acceleration")
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("[InsightFaceWrapper] Using GPU acceleration (CUDAExecutionProvider)")
        else:
            providers = ["CPUExecutionProvider"]
            print("[InsightFaceWrapper] Using CPU only (no GPU provider found)")

        ctx_id = 0 if device == "cuda" else -1

        # CHANGED: allow recognition too for direct embedding
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allowed_modules=["detection", "recognition"],  # ADDED
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        print(
            f"[InsightFaceWrapper] Loaded InsightFace with det_size={det_size} and providers={providers}"
        )

        # ADDED: optional toggles
        self._force_embed_only = os.getenv("FORCE_EMBED_ONLY", "0") == "1"

        # ADDED: try to access recognition model for direct embedding
        self._rec = None
        try:
            models = getattr(self.app, "models", None)
            if isinstance(models, dict):
                self._rec = models.get("recognition", None)
            if self._rec is None and hasattr(self.app, "model_zoo"):
                for m in getattr(self.app, "model_zoo", []):
                    if getattr(m, "taskname", "") == "recognition":
                        self._rec = m
                        break
        except Exception:
            self._rec = None

        # ADDED: determine recognition input size (fallback 112x112)
        self._rec_input = self.input_size
        try:
            ishape = getattr(self._rec, "input_size", None)
            if isinstance(ishape, (tuple, list)) and len(ishape) == 2:
                self._rec_input = (int(ishape[0]), int(ishape[1]))
        except Exception:
            pass

    # --------- unchanged detection path (camera etc.) ----------
    def detect_and_embed(self, frame: np.ndarray):
        faces = self.app.get(frame)
        results = []
        for f in faces:
            results.append(
                {
                    "bbox": np.array(f.bbox, dtype=np.int32),
                    "kps": f.kps if hasattr(f, "kps") else np.zeros((5, 2)),
                    "embedding": np.array(f.embedding, dtype=np.float32),
                }
            )
        return results

    # ---------- embedding-only on aligned/cropped face ----------
    def embed(self, bgr: np.ndarray) -> np.ndarray:
        """
        Return embedding from an ALREADY aligned + cropped face.
        No detection is run here.
        """
        if self._rec is not None:  # ADDED: direct recognition call
            try:
                W, H = self._rec_input
                face = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                if hasattr(self._rec, "get"):
                    emb = self._rec.get(face)
                elif hasattr(self._rec, "get_feat"):
                    emb = self._rec.get_feat(face)
                elif hasattr(self._rec, "forward"):
                    emb = self._rec.forward(face)
                else:
                    emb = None

                if emb is None:
                    return None
                return np.asarray(emb, dtype=np.float32).reshape(-1)
            except Exception:
                pass  # fall back below

        # CHANGED: final fallback — use app.get (detection) only if direct rec unavailable
        faces = self.app.get(bgr)
        if len(faces) > 0 and getattr(faces[0], "embedding", None) is not None:
            return np.array(faces[0].embedding, dtype=np.float32)
        return None

    def get_embedding(self, img_path: str) -> np.ndarray:
        frame = cv2.imread(img_path)
        if frame is None:
            raise ValueError(f"[InsightFaceWrapper] Cannot read image: {img_path}")
        # keep existing behavior (detection) for convenience callers
        faces = self.detect_and_embed(frame)
        if faces:
            return faces[0]["embedding"]
        return None
