import os
import sys
import importlib
import torch
from pathlib import Path

# -------------------------------------------------------------
# Correct project paths
# -------------------------------------------------------------
SRC_ROOT = Path(__file__).resolve().parent             # → Bachelor-Thesis-AI/src/
MODELS_DIR = SRC_ROOT / "models"                       # → src/models/
CONFIG_PATH = MODELS_DIR / "model.config"              # → src/models/model.config

# Ensure paths exist
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"model.config not found at {CONFIG_PATH}")

# Add project root and external/
PROJECT_ROOT = SRC_ROOT.parent                         # → Bachelor-Thesis-AI/
EXTERNAL_DIR = PROJECT_ROOT / "external"

for path in [str(SRC_ROOT), str(PROJECT_ROOT), str(EXTERNAL_DIR)]:
    if path not in sys.path:
        sys.path.append(path)

from models.wrap_facedetection import FaceDetectorAligner


# -------------------------------------------------------------
# Shared detector instance
# -------------------------------------------------------------
_DETECTOR = None

def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Loader] Initializing shared detector on: {device}")
        _DETECTOR = FaceDetectorAligner(device=device)
    return _DETECTOR


# -------------------------------------------------------------
# Model registry
# -------------------------------------------------------------
WRAPPERS = {
    "arcface": ("models.wrap_arcface", "ArcFaceWrapper"),
    "facenet": ("models.wrap_facenet", "FaceNetWrapper"),
    "adaface": ("models.wrap_adaface_onnx", "AdaFaceONNX"),
}


def available_models():
    return sorted(WRAPPERS.keys())


def load_model(model_name: str):
    """Dynamically load a model wrapper."""
    if model_name not in WRAPPERS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {', '.join(available_models())}"
        )

    module_name, class_name = WRAPPERS[model_name]

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Failed to import wrapper module '{module_name}': {e}")

    if not hasattr(module, class_name):
        raise AttributeError(f"Module '{module_name}' missing class '{class_name}'")

    wrapper_class = getattr(module, class_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Loader] Loading model '{model_name}' on device: {device}")

    wrapper = wrapper_class(device=device)

    # Attach shared face detector
    wrapper.detector = _get_detector()

    return wrapper
