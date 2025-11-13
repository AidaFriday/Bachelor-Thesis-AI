import os, sys, importlib, torch

# Make sure `external/` and project root are importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "external"))

from models.wrap_facedetection import FaceDetectorAligner

_DETECTOR = None


def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _DETECTOR = FaceDetectorAligner(device=device)
    return _DETECTOR


WRAPPERS = {
    "arcface": ("models.wrap_arcface", "ArcFaceWrapper"),
    "facenet": ("models.wrap_facenet", "FaceNetWrapper"),
    #"adaface": ("models.wrap_adaface", "AdaFaceWrapper"),  
    "adaface": ("models.wrap_adaface_onnx", "AdaFaceONNX"),

}


def available_models():
    return list(WRAPPERS.keys())


def load_model(model_name: str):
    if model_name not in WRAPPERS:
        raise ValueError(f"Unknown model: {model_name}")

    module_name, class_name = WRAPPERS[model_name]
    module = importlib.import_module(module_name)
    wrapper_class = getattr(module, class_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = wrapper_class(device=device)

    # ✅ Shared detector
    wrapper.detector = _get_detector()

    return wrapper
