import importlib
import torch


def load_model(model_name: str, config_path=None):
    """Load model wrapper from package (pip-available only)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wrappers = {
        "arcface": ("models.wrap_arcface", "ArcFaceWrapper"),
        "facenet": ("models.wrap_facenet", "FaceNetWrapper"),
        "insightface": ("models.wrap_insightface", "InsightFaceWrapper"),
    }
    if model_name not in wrappers:
        raise ValueError(f"Unknown model: {model_name}")

    module_name, class_name = wrappers[model_name]
    module = importlib.import_module(module_name)
    wrapper_class = getattr(module, class_name)

    return wrapper_class(device)
