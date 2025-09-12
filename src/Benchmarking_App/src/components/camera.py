import cv2
import numpy as np
import time
from collections import deque
from .face_detection import align_by_5pts


def run_camera_loop(app, encoder, out_size, choice: str):
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
        faces = app.get(frame)
        disp = frame.copy()

        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            kps = f.kps.astype(np.float32)

            crop = align_by_5pts(frame, kps, out_size=out_size)
            if crop is None:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, out_size)

            t1 = time.perf_counter()
            if choice == "1":
                emb = f.embedding
            else:
                emb = encoder.embed(crop)
            t2 = time.perf_counter()
            times_ms.append((t2 - t1) * 1000.0)

            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for (px, py) in kps.astype(int):
                cv2.circle(disp, (px, py), 2, (0, 255, 255), -1)

        now = time.perf_counter()
        fps = 1.0 / (now - last)
        last = now
        fps_hist.append(fps)

        y = 30
        if len(times_ms) > 5:
            p50 = np.percentile(np.array(times_ms), 50)
            p95 = np.percentile(np.array(times_ms), 95)
            cv2.putText(
                disp,
                f"{'buffalo_l' if choice=='1' else encoder.name} enc: {p50:.1f}/{p95:.1f} ms",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            y += 25
        cv2.putText(
            disp,
            f"FPS: {np.mean(fps_hist):.1f}    faces: {len(faces)}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Live FR Wrapper", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
