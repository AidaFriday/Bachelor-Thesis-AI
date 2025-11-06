import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List

from insightface.app import FaceAnalysis

# Standard ArcFace 112×112 reference points
REF_5PTS_112 = np.float32(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ]
)


def align_5pts(
    bgr: np.ndarray, kps: np.ndarray, out_size: Tuple[int, int]
) -> Optional[np.ndarray]:
    """Apply 5-point similarity transform to requested resolution."""
    ref = REF_5PTS_112.copy()
    scale = out_size[0] / 112.0
    ref *= scale

    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ref, method=cv2.LMEDS)
    if M is None:
        return None

    return cv2.warpAffine(bgr, M, out_size, flags=cv2.INTER_LINEAR)


class FaceDetectorAligner:

    def __init__(self, device="cpu"):
        ctx_id = 0 if device == "cuda" else -1

        # Load only detection + landmark models (no recognition)
        self.det = FaceAnalysis(
            name="buffalo_l", allowed_modules=["detection", "landmark"]
        )
        self.det.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Return raw detection with bounding boxes + 5-keypoints."""
        faces = self.det.get(frame)
        results = []
        for f in faces:
            results.append(
                {
                    "bbox": np.array(f.bbox, dtype=np.int32),
                    "kps": np.array(f.kps, dtype=np.float32),  # (5,2)
                }
            )
        return results

    def align_for(
        self, frame: np.ndarray, kps: np.ndarray, out_size=(112, 112)
    ) -> Optional[np.ndarray]:
        """Align face using 5 keypoints."""
        aligned = align_5pts(frame, kps, out_size=tuple(out_size))

        if aligned is None:
            # fallback: center crop
            h, w = frame.shape[:2]
            s = min(h, w)
            y0 = (h - s) // 2
            x0 = (w - s) // 2
            crop = frame[y0 : y0 + s, x0 : x0 + s]
            aligned = cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)

        return aligned.astype(np.uint8)

    # ✅ Compatibility method for all existing wrappers
    def get(self, frame: np.ndarray, out_size=(112, 112)) -> List[np.ndarray]:
        """
        Detects faces and returns aligned face crops (CHW=HWC BGR).
        This matches the expected output in ArcFace/Facenet/SphereFace wrappers.
        """
        detections = self.detect(frame)
        crops = []
        for d in detections:
            crop = self.align_for(frame, d["kps"], out_size)
            if crop is not None:
                crops.append(crop)
        return crops
