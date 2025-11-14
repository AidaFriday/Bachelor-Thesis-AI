import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


# ----------------------------------------------------------
# Reference 5 points for alignment (112 and 160)
# ----------------------------------------------------------
REF_5PTS_112 = np.float32([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
])

scale_160 = 160.0 / 112.0
REF_5PTS_160 = REF_5PTS_112 * scale_160


def align_face_5pts(bgr, kps, out_size=(160, 160)):
    ref = REF_5PTS_160 if out_size[0] == 160 else REF_5PTS_112
    M = cv2.estimateAffinePartial2D(kps.astype(np.float32), ref, method=cv2.LMEDS)[0]
    if M is None:
        return None
    return cv2.warpAffine(bgr, M, out_size, flags=cv2.INTER_LINEAR)


# ----------------------------------------------------------
# YOLOv5-face ONNX (GPU only)
# ----------------------------------------------------------
class YOLOv5FaceDetector:
    def __init__(self, onnx_path, conf_thres=0.5, iou_thres=0.4, providers=None):
        self.input_size = 640
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        if providers is None:
            providers = ["CPUExecutionProvider"]

        print(f"[YOLOv5-Face] Providers: {providers}")

        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, img):
        h, w = img.shape[:2]
        r = self.input_size / max(h, w)
        resized = cv2.resize(img, (int(w * r), int(h * r)))

        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized

        blob = canvas[:, :, ::-1].transpose(2, 0, 1)
        blob = blob.astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)
        return blob, r

    def detect(self, frame):
        blob, r = self._preprocess(frame)
        preds = self.session.run(None, {self.input_name: blob})[0][0]

        detections = []
        for det in preds:
            conf = det[4]
            if conf < self.conf_thres:
                continue

            x, y, w, h = det[:4]

            x1 = int(x / r - w / (2 * r))
            y1 = int(y / r - h / (2 * r))
            x2 = int(x / r + w / (2 * r))
            y2 = int(y / r + h / (2 * r))

            kps = np.array([
                [det[5] / r, det[6] / r],
                [det[7] / r, det[8] / r],
                [det[9] / r, det[10] / r],
                [det[11] / r, det[12] / r],
                [det[13] / r, det[14] / r],
            ], dtype=np.float32)

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "kps": kps,
                "conf": float(conf)
            })

        return detections


# ----------------------------------------------------------
# Unified detector + aligner
# ----------------------------------------------------------
class FaceDetectorAligner:
    def __init__(self, device="cpu"):
        ROOT = Path(__file__).resolve().parents[2]
        model_path = ROOT / "external" / "FaceNet_onnx" / "yolov5s-face.onnx"

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )

        print(f"[FaceDetectorAligner] Device={device}, Providers={providers}")

        self.detector = YOLOv5FaceDetector(model_path, providers=providers)

    def detect(self, frame):
        return self.detector.detect(frame)

    def align_for(self, frame, kps, out_size=(160, 160)):
        return align_face_5pts(frame, kps, out_size)
