import os
import sys
import shutil
import argparse
import cv2
import numpy as np

# Add project root for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)

from models.wrap_yolov5face import YOLOv5FaceDetector  # your code


# ---------- CONFIG ----------
YOLOV5_ONNX = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../../external/FaceNet_onnx/yolov5s-face.onnx",
    )
)

OUTPUT_DIR = "_CLEAN"
REJECTED_DIR = "_REJECTED"


# ---------- FACE CROPPING ----------
def crop_face(img, det):
    """Crop the face using YOLO bounding box."""
    x1, y1, x2, y2 = map(int, det["bbox"])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    face = img[y1:y2, x1:x2]
    return face


# ---------- MAIN CLEANING FUNCTION ----------
def clean_dataset(dataset_path):
    detector = YOLOv5FaceDetector(YOLOV5_ONNX)
    print("[INFO] YOLOv5-Face loaded successfully")

    out_clean = dataset_path + OUTPUT_DIR
    out_rej = dataset_path + REJECTED_DIR

    # Create output dirs
    os.makedirs(out_clean, exist_ok=True)
    os.makedirs(out_rej, exist_ok=True)

    kept = 0
    rejected = 0

    for person in os.listdir(dataset_path):
        folder = os.path.join(dataset_path, person)
        if not os.path.isdir(folder):
            continue

        # Output folders
        out_p_clean = os.path.join(out_clean, person)
        out_p_rej = os.path.join(out_rej, person)
        os.makedirs(out_p_clean, exist_ok=True)
        os.makedirs(out_p_rej, exist_ok=True)

        # Process each image
        for img_name in os.listdir(folder):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print("[REJECT:LOAD ERROR]", img_path)
                shutil.copy(img_path, out_p_rej)
                rejected += 1
                continue

            detections = detector.get(img)

            if len(detections) == 0:
                print("[REJECT:NO FACE]", img_path)
                shutil.copy(img_path, out_p_rej)
                rejected += 1
                continue

            # Pick the largest face
            detections.sort(
                key=lambda d: (d["bbox"][2] - d["bbox"][0])
                * (d["bbox"][3] - d["bbox"][1]),
                reverse=True,
            )
            best_det = detections[0]

            face = crop_face(img, best_det)

            if face is None or face.size == 0:
                print("[REJECT:CROP FAILED]", img_path)
                shutil.copy(img_path, out_p_rej)
                rejected += 1
                continue

            # Save cropped face
            out_img_path = os.path.join(out_p_clean, img_name)
            cv2.imwrite(out_img_path, face)
            kept += 1

            print("[OK:CROPPED]", img_path)

    print("\n---------------------------------------------")
    print("[DONE] Dataset cleaned with face cropping.")
    print(f"✓ Kept images: {kept}")
    print(f"✓ Rejected images: {rejected}")
    print(f"Cropped dataset saved to: {out_clean}")
    print(f"Rejected images saved to: {out_rej}")
    print("---------------------------------------------")


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    clean_dataset(args.dataset)
