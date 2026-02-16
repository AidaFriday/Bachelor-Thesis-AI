#camera.py
import cv2
import time
import numpy as np
from collections import deque


def run_camera_loop(wrapper):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print("Press ESC to exit.")
    fps_hist = deque(maxlen=60)
    last = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ✅ Use the UNIVERSAL detector
        detected = wrapper.detector.detect(frame)

        disp = frame.copy()
        faces = []

        for f in detected:
            bbox = f["bbox"]
            kps = f["kps"]

            # ✅ Align based on the model in use
            aligned = wrapper.detector.align_for(frame, wrapper.name)
            if aligned is None:
                continue

            emb = wrapper.embed(aligned)

            faces.append({"bbox": bbox, "kps": kps, "embedding": emb})

            x1, y1, x2, y2 = bbox
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for px, py in kps.astype(int):
                cv2.circle(disp, (px, py), 2, (0, 255, 255), -1)

        # ✅ FPS calculation
        now = time.perf_counter()
        fps = 1.0 / (now - last)
        last = now
        fps_hist.append(fps)

        cv2.putText(
            disp,
            f"{wrapper.name} | FPS: {np.mean(fps_hist):.1f} | faces: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Live FR Wrapper", disp)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
