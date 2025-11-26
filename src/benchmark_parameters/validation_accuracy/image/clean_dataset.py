import os
import cv2
import shutil
import argparse
import numpy as np
import sys


# FIX: Make "models" import work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)

from models.wrap_yolov5face import YOLOv5FaceDetector

# Path to yolov5s-face.onnx (adjust if needed)

YOLOV5_ONNX = (
    r"C:/programming/Bachelor-Thesis-AI/external/FaceNet_onnx/yolov5s-face.onnx"
)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
MIN_FACE_SIZE = 80  # minimum face width/height in pixels
MAX_FACES = 1  # only allow 1 face per image


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clean_dataset(dataset_path):
    print("[INFO] Loading YOLOv5-Face model...")
    detector = YOLOv5FaceDetector(YOLOV5_ONNX)

    rejected_root = os.path.join(dataset_path, "_REJECTED")
    ensure_dir(rejected_root)

    count_good = 0
    count_bad = 0

    print("[INFO] Starting dataset cleaning...\n")

    for person in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person)
        if not os.path.isdir(person_dir):
            continue
        if person == "_REJECTED":
            continue

        rejected_person_dir = os.path.join(rejected_root, person)
        ensure_dir(rejected_person_dir)

        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"[REJECT:IMREAD FAIL] {img_path}")
                shutil.move(img_path, rejected_person_dir)
                count_bad += 1
                continue

            # Detect faces
            dets = detector.get(img)

            # Reject if no faces
            if len(dets) == 0:
                print(f"[REJECT:NO FACE] {img_path}")
                shutil.move(img_path, rejected_person_dir)
                count_bad += 1
                continue

            # Reject if many faces
            if len(dets) > MAX_FACES:
                print(f"[REJECT:TOO MANY FACES] {img_path}")
                shutil.move(img_path, rejected_person_dir)
                count_bad += 1
                continue

            # Check size of detected face
            x1, y1, x2, y2 = dets[0]["bbox"]
            w = x2 - x1
            h = y2 - y1

            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                print(f"[REJECT:FACE TOO SMALL] {img_path} (w={w}, h={h})")
                shutil.move(img_path, rejected_person_dir)
                count_bad += 1
                continue

            # Otherwise keep
            count_good += 1

    print("\n-----------------------------------------------------")
    print("[DONE] Dataset cleaning complete!")
    print(f"✔ Kept images: {count_good}")
    print(f"✘ Rejected images: {count_bad}")
    print(f"Rejected images stored in: {rejected_root}")
    print("-----------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    clean_dataset(args.dataset)
