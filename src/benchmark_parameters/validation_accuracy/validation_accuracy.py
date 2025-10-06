import os, sys, time, json, argparse
import torch
import numpy as np
from PIL import Image


# ---------------- Helpers ----------------
def l2_normalize(E, axis=1, eps=1e-12):
    n = np.linalg.norm(E, axis=axis, keepdims=True)
    return E / np.maximum(n, eps)


def send_log(msg, level="info"):
    """Send a log message to GUI as JSON"""
    payload = {"log": msg, "level": level}
    print(json.dumps(payload))
    sys.stdout.flush()


def send_progress(done, total):
    """Send progress update to GUI"""
    payload = {"progress": done, "total": total}
    print(json.dumps(payload))
    sys.stdout.flush()


# ---------------- Embedding extractors ----------------
def get_test_embedding(model_name, model, mtcnn, img_path, device):
    if model_name == "facenet":
        img = Image.open(img_path).convert("RGB")
        face = mtcnn(img)
        if face is None:
            return None
        with torch.no_grad():
            emb = model(face.unsqueeze(0).to(device)).cpu().numpy().flatten()
        return l2_normalize(emb, axis=0)

    elif model_name in ["arcface", "insightface"]:
        faces = (
            model.get(img_path) if hasattr(model, "get") else model.get_faces(img_path)
        )
        if not faces:
            return None
        emb = getattr(
            faces[0], "embedding", getattr(faces[0], "normed_embedding", None)
        )
        return np.array(emb, dtype=np.float32) if emb is not None else None
    return None


def get_lfw_embedding(model_name, model, mtcnn, img_path, device):
    if model_name == "facenet":
        img = Image.open(img_path).convert("RGB").resize((160, 160))
        arr = np.array(img)
        img_tensor = torch.tensor(arr).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5
        with torch.no_grad():
            emb = model(img_tensor.unsqueeze(0).to(device)).cpu().numpy().flatten()
        return l2_normalize(emb, axis=0)

    elif model_name in ["arcface", "insightface"]:
        faces = (
            model.get(img_path) if hasattr(model, "get") else model.get_faces(img_path)
        )
        if not faces:
            return None
        emb = getattr(
            faces[0], "embedding", getattr(faces[0], "normed_embedding", None)
        )
        return np.array(emb, dtype=np.float32) if emb is not None else None
    return None


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="arcface | facenet | insightface")
    parser.add_argument("--dataset", type=str, help="Path to LFW dataset folder")
    parser.add_argument("--test-image", type=str, help="Single test image")
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()

    # settings.json fallback
    settings_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.json"
    )
    cfg = {}
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as f:
                cfg = json.load(f)
        except:
            pass

    model_name = (args.model or cfg.get("model", "facenet")).lower()
    LFW_PATH = args.dataset or cfg.get("dataset")
    TEST_IMAGE = args.test_image
    threshold = args.threshold

    if not LFW_PATH or not os.path.exists(LFW_PATH):
        print(json.dumps({"error": f"Invalid LFW dataset path: {LFW_PATH}"}))
        return
    if not TEST_IMAGE or not os.path.exists(TEST_IMAGE):
        print(json.dumps({"error": f"Invalid test image: {TEST_IMAGE}"}))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load model
    send_log(f"Loading model: {model_name}")
    mtcnn, model = None, None
    if model_name == "facenet":
        from facenet_pytorch import InceptionResnetV1, MTCNN

        mtcnn = MTCNN(image_size=160, margin=0, device=device)
        model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    elif model_name == "arcface":
        import insightface

        model = insightface.model_zoo.get_model("arcface_r100_v1")
        model.prepare(ctx_id=0)
    elif model_name == "insightface":
        from insightface.app import FaceAnalysis

        model = FaceAnalysis(name="antelopev2")
        model.prepare(ctx_id=0)
    else:
        print(json.dumps({"error": f"Unknown model: {model_name}"}))
        return

    # ---- embeddings
    send_log(f"Extracting embedding for test image: {TEST_IMAGE}")
    q_emb = get_test_embedding(model_name, model, mtcnn, TEST_IMAGE, device)
    if q_emb is None:
        print(json.dumps({"error": "No face found in test image"}))
        return

    # Determine the identity folder of test image
    test_person = os.path.basename(os.path.dirname(TEST_IMAGE))

    positives, negatives = [], []
    t0 = time.time()
    n_files = 0

    # Pre-count images for progress bar
    total_images = sum(
        1
        for root, _, files in os.walk(LFW_PATH)
        for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    send_log(f"Scanning dataset at {LFW_PATH}")
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

            # Send progress every 100 images (or for each image if small dataset)
            if n_files % 100 == 0 or n_files == total_images:
                send_progress(n_files, total_images)

    dt = time.time() - t0
    send_log(
        f"Finished scanning {n_files} images in {dt:.2f}s ({n_files/dt:.1f} img/s)"
    )

    # ---- final JSON payload for GUI plots
    payload = {
        "model": model_name,
        "dataset": LFW_PATH,
        "threshold": threshold,
        "positives": positives,
        "negatives": negatives,
    }
    print(json.dumps(payload))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
