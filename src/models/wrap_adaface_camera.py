# models/wrap_adaface_camera.py

from models.wrap_adaface import AdaFaceWrapper


# Re-export for connector.py
class AdaFaceCameraWrapper(AdaFaceWrapper):
    name = "adaface_camera"
