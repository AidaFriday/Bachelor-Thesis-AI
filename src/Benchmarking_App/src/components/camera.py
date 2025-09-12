import cv2
import time
import numpy as np
from collections import deque
from .face_detection import align_by_5pts


def run_camera_loop(wrapper):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print("Press ESC to exit.")
    times_ms = deque(maxlen=200)
    fps_hist = deque(maxlen=60)
    last = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = wrapper.detect_and_embed(frame)
        disp = frame.copy()

        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            emb = f["embedding"]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for (px, py) in f["kps"].astype(int):
                cv2.circle(disp, (px, py), 2, (0, 255, 255), -1)

        now = time.perf_counter()
        fps = 1.0 / (now - last)
        last = now
        fps_hist.append(fps)

        cv2.putText(disp, f"{wrapper.name} | FPS: {np.mean(fps_hist):.1f} | faces: {len(faces)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Live FR Wrapper", disp)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
