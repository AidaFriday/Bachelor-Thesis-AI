# models/wrap_adaface_camera.py

from models.wrap_adaface import AdaFaceWrapper
import numpy as np


class AdaFaceCameraWrapper(AdaFaceWrapper):
    name = "adaface_camera"

    # Override embed to ensure flat embeddings
    def embed(self, img_bgr):
        emb = super().embed(img_bgr)
        if emb is None:
            return None

        # ALWAYS flatten → (512,) instead of (1, 512)
        return np.asarray(emb, dtype=np.float32).reshape(-1)
