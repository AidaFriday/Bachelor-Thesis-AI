import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Reference landmarks for alignment
REF_5PTS_112 = np.float32([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
])
scale_160 = 160.0 / 112.0
REF_5PTS_160 = REF_5PTS_112 * scale_160

def align_by_5pts(bgr: np.ndarray, kps: np.ndarray, out_size=(112, 112)) -> np.ndarray:
    ref = REF_5PTS_112 if out_size == (112, 112) else REF_5PTS_160
    M = cv2.estimateAffinePartial2D(kps.astype(np.float32), ref, method=cv2.LMEDS)[0]
    if M is None:
        return None
    return cv2.warpAffine(bgr, M, out_size, flags=cv2.INTER_LINEAR)


class RetinaFaceDetector:
    """Wrapper for RetinaFace-based face detection (from insightface)."""
    def __init__(self, device: str = "cpu", det_size=(640, 640)):
        ctx_id = 0 if device.lower().startswith("cuda") else -1
        # initialize RetinaFace via insightface
        self.app = FaceAnalysis(name="retinaface_r50", allowed_modules=["detection"])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
    
    def get(self, frame: np.ndarray):
        """
        Detect faces in frame.

        Returns list of face objects with:
            - bbox (x1, y1, x2, y2)
            - kps: landmarks (5 points)
            - (if available) confidence
        """
        faces = self.app.get(frame)
        return faces
