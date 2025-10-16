# ==== validation_accuracy.py ====
import os
import sys
import time
import json
import argparse
import shutil
import torch
import numpy as np
import cv2
from PIL import Image


# ---------------------------------------------------------------------
# 🔹 Helper Functions
# ---------------------------------------------------------------------
def l2_normalize(E, axis=1, eps=1e-12):
    """L2-normalize embeddings along given axis."""
    n = np.linalg.norm(E, axis=axis, keepdims=True)
    return E / np.maximum(n, eps)


def send_log(msg, level="info"):
    """Send structured log message for GUI/CLI."""
    payload = {"log": msg, "level": level}
    print(json.dumps(payload))
    sys.stdout.flush()


def send_progress(done, total):
    """Send progress updates as JSON payload."""
    pct = (done / total) * 100 if total > 0 else 0
    payload = {"progress": done, "total": total, "percent": round(pct, 2)}
    print(json.dumps(payload))
    sys.stdout.flush()


# ---------------------------------------------------------------------
# 🔹 Embedding Extractors
# ---------------------------------------------------------------------
def load_image_for_insightface(img_path):
    """Load image as numpy array for InsightFace (BGR)."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {img_path}")
    return img


def get_test_embedding(model_name, model, mtcnn, img_path, device):
    """Extract normalized embedding for the test image."""
    if model_name == "facenet":
        img = Image.open(img_path).convert("RGB")
        face = mtcnn(img)
        if face is None:
            return None
        with torch.no_grad():
            emb = model(face.unsqueeze(0).to(device)).cpu().numpy().flatten()
        return l2_normalize(emb, axis=0)

    elif model_name in ["arcface", "insightface"]:
        img = load_image_for_insightface(img_path)
        faces = model.get(img)
        if not faces:
            return None
        emb = getattr(
            faces[0], "embedding", getattr(faces[0], "normed_embedding", None)
        )
        return np.array(emb, dtype=np.float32) if emb is not None else None

    return None


def get_lfw_embedding(model_name, model, mtcnn, img_path, device):
    """Extract embedding for an LFW dataset image."""
    if model_name == "facenet":
        img = Image.open(img_path).convert("RGB").resize((160, 160))
        arr = np.array(img)
        img_tensor = torch.tensor(arr).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5
        with torch.no_grad():
            emb = model(img_tensor.unsqueeze(0).to(device)).cpu().numpy().flatten()
        return l2_normalize(emb, axis=0)

    elif model_name in ["arcface", "insightface"]:
        img = load_image_for_insightface(img_path)
        faces = model.get(img)
        if not faces:
            return None
        emb = getattr(
            faces[0], "embedding", getattr(faces[0], "normed_embedding", None)
        )
        return np.array(emb, dtype=np.float32) if emb is not None else None

    return None


# ---------------------------------------------------------------------
# 🔹 InsightFace Model Loader with Auto-Healing
# ---------------------------------------------------------------------
def load_insightface_safe():
    """Load and prepare InsightFace (buffalo_l) model with self-healing."""
    from insightface.app import FaceAnalysis

    send_log("Initializing InsightFace (buffalo_l)...")

    model = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
    try:
        model.prepare(ctx_id=0, det_size=(640, 640))
        return model
    except AssertionError:
        send_log(
            "[WARN] InsightFace model missing or corrupt. Re-downloading...",
            level="warn",
        )
        cache_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            send_log(f"Deleted corrupt model cache: {cache_dir}")
        model = FaceAnalysis(
            name="buffalo_l", allowed_modules=["detection", "recognition"]
        )
        model.prepare(ctx_id=0, det_size=(640, 640))
        send_log("✅ InsightFace (buffalo_l) reinitialized successfully.")
        return model


# ---------------------------------------------------------------------
# 🔹 Main Benchmark Logic
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run face verification benchmark.")
    parser.add_argument("--model", type=str, help="arcface | facenet | insightface")
    parser.add_argument("--dataset", type=str, help="Path to LFW dataset folder")
    parser.add_argument("--test-image", type=str, help="Single test image path")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--iters", type=int, default=None, help="(ignored, for compatibility)"
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Load settings.json (if available)
    # -----------------------------------------------------------------
    settings_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "settings.json",
    )

    cfg = {}
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as f:
                cfg = json.load(f)
        except Exception:
            pass

    model_name = (args.model or cfg.get("model", "facenet")).lower()
    LFW_PATH = args.dataset or cfg.get("dataset")
    TEST_IMAGE = args.test_image
    threshold = args.threshold

    # -----------------------------------------------------------------
    # Validate input paths
    # -----------------------------------------------------------------
    if not LFW_PATH or not os.path.exists(LFW_PATH):
        print(json.dumps({"error": f"Invalid LFW dataset path: {LFW_PATH}"}))
        sys.exit(1)
    if not TEST_IMAGE or not os.path.exists(TEST_IMAGE):
        print(json.dumps({"error": f"Invalid test image: {TEST_IMAGE}"}))
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------
    # Load selected model
    # -----------------------------------------------------------------
    send_log(f"Loading model: {model_name}")
    mtcnn, model = None, None

    if model_name == "facenet":
        from facenet_pytorch import InceptionResnetV1, MTCNN

        mtcnn = MTCNN(image_size=160, margin=0, device=device)
        model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    elif model_name == "arcface":
        send_log("Using InsightFace (buffalo_l) for ArcFace mode...")
        from insightface.app import FaceAnalysis

        model = FaceAnalysis(
            name="buffalo_l", allowed_modules=["detection", "recognition"]
        )
        model.prepare(ctx_id=0, det_size=(640, 640))

    elif model_name == "insightface":
        model = load_insightface_safe()

    else:
        print(json.dumps({"error": f"Unknown model: {model_name}"}))
        sys.exit(1)

    # -----------------------------------------------------------------
    # Extract embedding for the query/test image
    # -----------------------------------------------------------------
    send_log(f"Extracting embedding for test image: {TEST_IMAGE}")
    q_emb = get_test_embedding(model_name, model, mtcnn, TEST_IMAGE, device)
    if q_emb is None:
        print(json.dumps({"error": "No face detected in test image"}))
        sys.exit(1)

    test_person = os.path.basename(os.path.dirname(TEST_IMAGE))

    positives, negatives = [], []
    t0 = time.time()
    n_files = 0

    # Pre-count all valid images for progress tracking
    total_images = sum(
        1
        for root, _, files in os.walk(LFW_PATH)
        for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    send_log(f"Scanning dataset at {LFW_PATH} ({total_images} images)")

    last_percent_logged = -1
    for root, _, files in os.walk(LFW_PATH):
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(root, f)
            emb = get_lfw_embedding(model_name, model, mtcnn, path, device)
            if emb is None:
                continue

            sim = float(np.dot(q_emb, emb))
            person = os.path.basename(os.path.dirname(path))
            if person == test_person:
                positives.append(sim)
            else:
                negatives.append(sim)

            n_files += 1

            # update progress every 1% or every 100 images
            percent = int((n_files / total_images) * 100)
            if (
                percent > last_percent_logged
                or n_files % 100 == 0
                or n_files == total_images
            ):
                last_percent_logged = percent
                send_progress(n_files, total_images)

    dt = time.time() - t0
    send_log(
        f"✅ Finished scanning {n_files} images in {dt:.2f}s "
        f"({n_files / max(dt, 1e-5):.1f} images/sec)"
    )

    # -----------------------------------------------------------------
    # Emit final payload
    # -----------------------------------------------------------------
    payload = {
        "model": model_name,
        "dataset": LFW_PATH,
        "threshold": threshold,
        "positives": positives,
        "negatives": negatives,
        "elapsed_sec": round(dt, 2),
        "count": n_files,
    }

    print(json.dumps(payload))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
