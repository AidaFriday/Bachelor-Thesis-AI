import importlib
import torch

# ✅ Add this import
from models.wrap_facedetection import FaceDetectorAligner

# ✅ Keep one shared detector instance for all models (faster)
_DETECTOR = None


def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _DETECTOR = FaceDetectorAligner(device=device)
    return _DETECTOR


def load_model(model_name: str, config_path=None):
    """Load model wrapper and attach the shared face detector/aligner."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrappers = {
        "arcface": ("models.wrap_arcface", "ArcFaceWrapper"),
        "facenet": ("models.wrap_facenet", "FaceNetWrapper"),
        "insightface": ("models.wrap_insightface", "InsightFaceWrapper"),
        "sphereface": ("models.wrap_sphereface", "SphereFaceWrapper"),
        "lightcnn": ("models.wrap_lightcnn", "LightCNNWrapper"),
        "deepface": ("models.wrap_deepface", "DeepFaceWrapper"),
    }

    if model_name not in wrappers:
        raise ValueError(f"Unknown model: {model_name}")

    module_name, class_name = wrappers[model_name]
    module = importlib.import_module(module_name)
    wrapper_class = getattr(module, class_name)

    # ✅ Create embedding model
    wrapper = wrapper_class(device)

    # ✅ Attach detector once (shared)
    wrapper.detector = _get_detector()

    return wrapper
