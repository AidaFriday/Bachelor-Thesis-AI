import os, sys, time, math, json
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from pathlib import Path

# ---- DEBUG LOGGER ----
import logging

logging.basicConfig(
    level=logging.DEBUG,  # change to INFO when done
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("valacc-debug")

# dataset path
LFW_PATH = r"C:\programming\Datasets\LFW\lfw-deepfunneled"
TEST_IMAGE = r"C:\programming\Bachelor-Thesis-AI\src\benchmark_parameters\validation_accuracy\test_image\test_image1.jpg"

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face detector (MTCNN) and embedding model (ResNet)
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)


# ---------------- helpers ----------------
def _np(v):
    try:
        return v.detach().cpu().numpy()
    except Exception:
        return np.asarray(v)


def dbg_env(model=None, device=None):
    try:
        import torch as _t

        log.debug(f"torch.__version__={_t.__version__}")
        log.debug(f"CUDA available={_t.cuda.is_available()}")
        if _t.cuda.is_available():
            log.debug(f"CUDA device={_t.cuda.get_device_name(0)}")
    except Exception as e:
        log.warning(f"Could not query torch env: {e}")

    if model is not None:
        try:
            dim = getattr(model, "last_linear", None)
            emb_dim = 512 if dim is not None else None
            log.debug(
                f"Model: {model.__class__.__name__}, expected embedding_dim≈{emb_dim}"
            )
        except Exception as e:
            log.debug(f"Model info error: {e}")
    if device is not None:
        log.debug(f"Device: {device}")


def dbg_image(name, img_tensor):
    x = _np(img_tensor)
    log.debug(
        f"[{name}] shape={tuple(x.shape)}, dtype={x.dtype}, "
        f"min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}"
    )


def dbg_embedding(name, emb, do_stats=True):
    e = _np(emb).astype(np.float32)
    if e.ndim == 1:
        e = e[None, :]
    norms = np.linalg.norm(e, axis=1)
    log.debug(f"[{name}] shape={e.shape}, first_norm={norms[0]:.6f}")
    if do_stats:
        log.debug(
            f"[{name}] norms→ min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}"
        )
    if np.any(~np.isfinite(e)):
        log.error(f"[{name}] has NaN/Inf values!")
    return e


def l2_normalize(E, axis=1, eps=1e-12):
    n = np.linalg.norm(E, axis=axis, keepdims=True)
    return E / np.maximum(n, eps)


def cosine_sim(q, E):
    q = q.reshape(1, -1)
    return (q @ E.T).ravel()  # assumes both are L2-normalized


def topk(sim, k=5):
    idx = np.argsort(-sim)[:k]
    return idx, sim[idx]


def explain_no_match(sim, ids, threshold):
    mx = float(sim.max()) if sim.size else float("-inf")
    msg = f"No match because max_sim={mx:.4f} < threshold={threshold:.4f}"
    near = np.argsort(-sim)[:5]
    hint = [(int(i), float(sim[i]), ids[i] if ids else None) for i in near]
    return msg, hint


# --- For the test image (may not be cropped) ---
def get_test_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    face = mtcnn(img)  # returns CHW tensor normalized to [-1,1] for facenet-pytorch
    if face is None:
        print(f"❌ No face detected in test image: {img_path}")
        return None
    dbg_image("test_face_tensor(pre-matmul)", face)
    with torch.no_grad():
        emb = model(face.unsqueeze(0).to(device)).cpu().float().numpy().flatten()
    dbg_embedding("test_embedding_raw", emb)
    emb = l2_normalize(emb, axis=0)  # ensure unit vector
    dbg_embedding("test_embedding_normed", emb)
    return emb


# --- For LFW dataset images (already cropped/aligned to ~250x250) ---
# We keep a simple resize->[-1,1] normalization to 160x160 to match the test pipeline.
def get_lfw_embedding(img_path):
    img = Image.open(img_path).convert("RGB").resize((160, 160))
    # torchvision-like normalization to [-1,1]
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5
    dbg_image("lfw_img_tensor(pre-matmul)", img_tensor)
    with torch.no_grad():
        emb = model(img_tensor.unsqueeze(0).to(device)).cpu().float().numpy().flatten()
    dbg_embedding("lfw_embedding_raw", emb, do_stats=False)
    emb = l2_normalize(emb, axis=0)
    return emb


# ---------------- main ----------------
if __name__ == "__main__":
    dbg_env(model, device)

    print("🔎 Extracting test image embedding...")
    test_embedding = get_test_embedding(TEST_IMAGE)
    if test_embedding is None:
        sys.exit(0)

    print("🔎 Searching in LFW dataset...")
    threshold = 0.50  # start conservative; adjust after seeing Top-K

    best_sim = -1.0
    best_path = None
    sims = []
    ids = []

    n_files = 0
    t0 = time.time()

    for root, _, files in os.walk(LFW_PATH):
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            lfw_path = os.path.join(root, f)
            try:
                emb = get_lfw_embedding(lfw_path)
            except Exception as e:
                log.warning(f"Skipping {lfw_path}: {e}")
                continue

            # cosine similarity (embeddings already unit-normalized)
            sim = float(np.dot(test_embedding, emb))
            sims.append(sim)
            # use the parent folder name (identity) as id for readability
            ids.append(
                os.path.basename(os.path.dirname(lfw_path))
                + "/"
                + os.path.basename(lfw_path)
            )

            n_files += 1
            if sim > best_sim:
                best_sim = sim
                best_path = lfw_path

            if n_files % 1000 == 0:
                log.debug(
                    f"Processed {n_files} images… current best={best_sim:.4f} ({best_path})"
                )

    dt = time.time() - t0
    if n_files == 0:
        log.error("No images found in LFW_PATH. Check the path.")
        print("❌ No match in LFW dataset.")
        sys.exit(0)

    sims = np.asarray(sims, dtype=np.float32)
    log.info(f"Scanned {n_files} LFW images in {dt:.2f}s → {n_files/dt:.1f} img/s")
    log.debug(
        f"Similarity range: min={sims.min():.4f}, max={sims.max():.4f}, mean={sims.mean():.4f}"
    )

    # Top-K dump
    K = 5
    top_idx, top_vals = topk(sims, k=K)
    log.info("Top-{} matches:".format(K))
    for rank, (i, val) in enumerate(zip(top_idx, top_vals), start=1):
        log.info(f"  #{rank}: {ids[i]}  cos={val:.4f}")

    # Decision
    if best_sim >= threshold:
        print(
            f"✅ Match found: {best_path} (similarity={best_sim:.3f}, threshold={threshold:.2f})"
        )
    else:
        msg, hint = explain_no_match(sims, ids, threshold)
        log.warning(msg)
        for i, s, pid in hint:
            log.debug(f" near[{i}]  {pid}  cos={s:.4f}")
        print("❌ No match in LFW dataset.")
