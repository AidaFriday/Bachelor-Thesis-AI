import cv2
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm


import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]  # → src/
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# =====================================================
# PROJECT IMPORTS (SAME AS LIVE FEED)
# =====================================================
from connector import load_model
from models.wrap_facedetection import FaceDetectorAligner

# =====================================================
# PATHS
# =====================================================
SCRIPT_DIR = Path(__file__).parent
LOG_FILE = SCRIPT_DIR / "emb_extraction.log"

DATASET_ROOT = Path(r"C:\programming\Datasets\CUSTOM_DATASET_ORG")

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("emb_extraction")

# =====================================================
# EMBEDDING MODELS (MUST MATCH LIVE FEED)
# =====================================================
EMBEDDERS = {
    "arcface": {
        "loader_name": "arcface",
        "output": "faces_arcface.npz",
    },
    "facenet": {
        "loader_name": "facenet_camera",
        "output": "faces_facenet.npz",
    },
    "adaface": {
        "loader_name": "adaface_camera",
        "output": "faces_adaface.npz",
    },
}

# =====================================================
# FILTERING
# =====================================================
MIN_FACE_SIZE = 64
MIN_KPS_CONF = 5  # number of landmarks required

# =====================================================
# HELPERS
# =====================================================
def valid_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) >= MIN_FACE_SIZE and (y2 - y1) >= MIN_FACE_SIZE


def normalize(emb: np.ndarray) -> np.ndarray:
    return emb / np.linalg.norm(emb)


# =====================================================
# MAIN
# =====================================================
def main():
    if not DATASET_ROOT.exists():
        raise RuntimeError(f"Dataset not found: {DATASET_ROOT}")

    identities = sorted(p for p in DATASET_ROOT.iterdir() if p.is_dir())
    log.info(f"Found {len(identities)} identities")

    # -------------------------------------------------
    # 1️⃣ SHARED DETECTOR (EXACT SAME AS LIVE)
    # -------------------------------------------------
    log.info("Initializing shared face detector")
    detector = FaceDetectorAligner(device="cpu")

    # -------------------------------------------------
    # 2️⃣ PROCESS EACH EMBEDDING MODEL
    # -------------------------------------------------
    for embed_name, cfg in EMBEDDERS.items():
        log.info("=" * 60)
        log.info(f"Building dataset for model: {embed_name}")
        log.info("=" * 60)

        wrapper = load_model(cfg["loader_name"])
        log.info(f"Loaded embedder: {wrapper.name}")

        all_embeddings = []
        all_labels = []
        all_paths = []

        for person_dir in tqdm(identities, desc=f"People ({embed_name})"):
            label = person_dir.name
            images = sorted(person_dir.glob("*.*"))

            person_faces = 0

            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                detections = detector.detect(img)

                for det in detections:
                    bbox = det["bbox"]
                    kps = det["kps"]

                    if not valid_bbox(bbox):
                        continue
                    if kps is None or len(kps) < MIN_KPS_CONF:
                        continue

                    aligned = detector.align_for(img, kps)
                    if aligned is None:
                        continue

                    emb = wrapper.embed(aligned)
                    if emb is None:
                        continue

                    emb = normalize(emb.astype(np.float32))

                    all_embeddings.append(emb)
                    all_labels.append(label)
                    all_paths.append(str(img_path))
                    person_faces += 1

            if person_faces == 0:
                log.warning(f"Identity '{label}' has 0 usable faces")

        if not all_embeddings:
            log.warning(f"No embeddings collected for {embed_name}")
            continue

        embeddings = np.vstack(all_embeddings)
        labels = np.array(all_labels)
        paths = np.array(all_paths)

        output_path = SCRIPT_DIR / cfg["output"]
        np.savez_compressed(
            output_path,
            embeddings=embeddings,
            labels=labels,
            paths=paths,
        )

        log.info(
            f"Saved {len(embeddings)} embeddings → {output_path.name}"
        )

    log.info("=" * 60)
    log.info("DONE | All embedding databases created")
    log.info("=" * 60)


# =====================================================
# ENTRY
# =====================================================
if __name__ == "__main__":
    main()
