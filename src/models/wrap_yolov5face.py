#wrap_yolov5face.py
import cv2
import numpy as np
import onnxruntime as ort


class YOLOv5FaceDetector:
    def __init__(self, onnx_path, conf_thres=0.5, iou_thres=0.4, input_size=640, providers=None):
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    # ----------------------------------------------------------
    # PREPROCESS
    # ----------------------------------------------------------
    def _preprocess(self, img):
        h, w = img.shape[:2]
        r = self.input_size / max(h, w)
        resized = cv2.resize(img, (int(w * r), int(h * r)))

        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized

        blob = canvas[:, :, ::-1].transpose(2, 0, 1)
        blob = np.expand_dims(blob, 0).astype(np.float32) / 255.0
        return blob, r

    # ----------------------------------------------------------
    # INFERENCE
    # ----------------------------------------------------------
    def get(self, frame):
        blob, r = self._preprocess(frame)
        output = self.session.run(None, {self.input_name: blob})[0]

        # YOLOv5-face output = (1, N, 15)
        preds = output[0]

        detections = []

        for det in preds:
            conf = det[4]
            if conf < self.conf_thres:
                continue

            x, y, w, h = det[:4]
            x /= r; y /= r; w /= r; h /= r

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # 5 landmarks
            kps = np.array([
                [det[5] / r, det[6] / r],
                [det[7] / r, det[8] / r],
                [det[9] / r, det[10] / r],
                [det[11] / r, det[12] / r],
                [det[13] / r, det[14] / r]
            ], dtype=np.float32)

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "kps": kps,
                "conf": float(conf)
            })

        return detections
