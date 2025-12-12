# wrap_facedetection.py
import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ----------------------------------------------------------
# Reference 5 points for alignment (112 and 160)
# ----------------------------------------------------------
REF_5PTS_112 = np.float32(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ]
)

scale_160 = 160.0 / 112.0
REF_5PTS_160 = REF_5PTS_112 * scale_160


def align_face_5pts(bgr, kps, out_size=(160, 160)):
    ref = REF_5PTS_160 if out_size[0] == 160 else REF_5PTS_112
    M, _ = cv2.estimateAffinePartial2D(
        kps.astype(np.float32), ref, method=cv2.LMEDS
    )
    if M is None:
        return None
    return cv2.warpAffine(bgr, M, out_size, flags=cv2.INTER_LINEAR)


# ----------------------------------------------------------
# SCRFD detector wrapper (InsightFace)
# ----------------------------------------------------------
class SCRFDFaceDetector:
    def __init__(self, device="cpu", det_size=(640, 640)):
        ctx_id = 0 if device.startswith("cuda") else -1

        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect(self, frame):
        faces = self.app.get(frame)

        detections = []
        for f in faces:
            bbox = f.bbox.astype(int)
            kps = f.kps.astype(np.float32)

            detections.append(
                {
                    "bbox": (bbox[0], bbox[1], bbox[2], bbox[3]),
                    "kps": kps,
                    "conf": float(f.det_score),
                }
            )

        return detections


# ----------------------------------------------------------
# Unified detector + aligner (UNCHANGED API)
# ----------------------------------------------------------
class FaceDetectorAligner:
    def __init__(self, device="cpu"):
        print(f"[FaceDetectorAligner] Device={device}, Backend=SCRFD")
        self.detector = SCRFDFaceDetector(device=device)

    def detect(self, frame):
        return self.detector.detect(frame)

    def align_for(self, frame, kps, out_size=(160, 160)):
        return align_face_5pts(frame, kps, out_size)
