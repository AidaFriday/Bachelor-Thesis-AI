import os
import json
import importlib
import torch


def load_model(model_name: str, config_path=None):
    """Load model wrapper based on config file."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve project root (the "src" directory)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if config_path is None:
        config_path = os.path.join(base_dir, "models", "model.config")

    # --- Load config ---
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    if model_name not in config:
        raise ValueError(f"Model '{model_name}' not in config")

    model_cfg = config[model_name]
    wrapper_path = model_cfg["wrapper"]
    module_name, class_name = wrapper_path.split(".")
    module = importlib.import_module(f"models.{module_name}")
    wrapper_class = getattr(module, class_name)

    # Path can be None (e.g. ArcFace auto-downloads its models)
    model_path = model_cfg.get("path", None)
    if model_path:
        # Always resolve relative to src/
        model_path = os.path.join(base_dir, model_path)

    input_size = tuple(model_cfg["input_size"])

    return wrapper_class(device, model_path, input_size)
