# connector.py
import os
import sys
import importlib
import torch
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent
EXTERNAL_DIR = PROJECT_ROOT / "external"

for p in (SRC_ROOT, PROJECT_ROOT, EXTERNAL_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.append(sp)

from models.wrap_facedetection import FaceDetectorAligner

# --- Model registry (hardcoded). Remove model.config check since unused.
WRAPPERS = {
    "arcface": ("models.wrap_arcface", "ArcFaceWrapper"),
    "facenet": ("models.wrap_facenet", "FaceNetWrapper"),
    "adaface": ("models.wrap_adaface_onnx", "AdaFaceONNX"),
}


def available_models():
    return sorted(WRAPPERS.keys())


# Shared detector, but keyed by device so the detector matches the model device.
_DETECTORS = {}


def _get_detector(device: str):
    dev = device.lower()
    det = _DETECTORS.get(dev)
    if det is None:
        print(f"[Loader] Initializing shared detector on: {dev}")
        _DETECTORS[dev] = det = FaceDetectorAligner(device=dev)
    return det


def load_model(model_name: str, device: str | None = None):
    """
    Load a model wrapper.
    device:
      - None / "auto": use CUDA if available else CPU
      - "cpu" or "cuda" (or "cuda:0")
    """
    if model_name not in WRAPPERS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {', '.join(available_models())}"
        )

    module_name, class_name = WRAPPERS[model_name]

    # Resolve device
    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import wrapper module '{module_name}': {e}"
        ) from e

    if not hasattr(module, class_name):
        raise AttributeError(f"Module '{module_name}' missing class '{class_name}'")

    wrapper_class = getattr(module, class_name)
    print(f"[Loader] Loading model '{model_name}' on device: {device}")

    # Nice error if onnxruntime is missing for AdaFaceONNX
    if model_name == "adaface":
        try:
            import onnxruntime  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "AdaFace (ONNX) requires onnxruntime/onnxruntime-gpu. "
                "Install with: pip install onnxruntime-gpu  (or onnxruntime for CPU)."
            ) from e

    wrapper = wrapper_class(device=device)

    # Attach a detector on the same device as the wrapper
    wrapper.detector = _get_detector(device)

    # Encourage consistent naming
    if not hasattr(wrapper, "name"):
        wrapper.name = model_name

    return wrapper
