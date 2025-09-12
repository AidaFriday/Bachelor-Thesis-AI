import os
import json
import importlib
import torch


def load_model(model_name: str, config_path="models/model.config"):
    """Load model wrapper based on config file."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config_path, "r") as f:
        config = json.load(f)

    if model_name not in config:
        raise ValueError(f"Model '{model_name}' not in config")

    model_cfg = config[model_name]
    wrapper_path = model_cfg["wrapper"]
    module_name, class_name = wrapper_path.split(".")
    module = importlib.import_module(f"models.{module_name}")
    wrapper_class = getattr(module, class_name)

    return wrapper_class(device, model_cfg["path"], tuple(model_cfg["input_size"]))
