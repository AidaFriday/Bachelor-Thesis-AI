# dataset/manager.py

import os
from typing import Dict, Type
from dataset.LFW import list_all_images as lfw_list
from dataset.YTF import list_all_images as ytf_list

# You’ll add new ones here as you grow
DATASET_REGISTRY: Dict[str, Dict[str, object]] = {
    "lfw": {
        "loader": lfw_list,
        "default_path": r"C:\programming\Datasets\LFW\lfw-deepfunneled",
        "kind": "image",
    },
    "ytf": {
        "loader": ytf_list,
        "default_path": r"C:\programming\Datasets\YTF\aligned_images_DB",
        "kind": "video",
    },
    # future example
    # "celeba": { "loader": celeba_list, "default_path": "C:/Datasets/CelebA", "kind": "image" },
}


class DatasetManager:
    def __init__(self):
        self.current_name = None
        self.current_path = None

    def list_available(self):
        return list(DATASET_REGISTRY.keys())

    def set_dataset(self, name: str, path: str = None):
        name = name.lower()
        if name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {name}")

        entry = DATASET_REGISTRY[name]
        self.current_name = name
        self.current_path = path or entry["default_path"]

        if not os.path.exists(self.current_path):
            raise FileNotFoundError(f"Dataset path not found: {self.current_path}")

    def get_info(self):
        if not self.current_name:
            return None
        entry = DATASET_REGISTRY[self.current_name]
        return {
            "name": self.current_name,
            "path": self.current_path,
            "kind": entry["kind"],
        }

    def load_images(self, limit=None, shuffle=True, verbose=False):
        if not self.current_name:
            raise RuntimeError("Dataset not set yet")
        loader = DATASET_REGISTRY[self.current_name]["loader"]
        return loader(self.current_path, limit=limit, shuffle=shuffle, verbose=verbose)
