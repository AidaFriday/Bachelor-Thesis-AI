import cv2
import logging
from pathlib import Path
from insightface.app import FaceAnalysis
from tqdm import tqdm

# ==============================
# PATHS
# ==============================
SCRIPT_DIR = Path(__file__).parent
LOG_FILE = SCRIPT_DIR / "emb_extraction.log"

DATASET_ROOT = Path(r"C:\programming\Datasets\CUSTOM_DATASET_ORG")

# ==============================
# LOGGING (FILE + CONSOLE)
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

log = logging.getLogger("emb_extraction")

log.info("=== Face Extraction Started ===")
log.info(f"Dataset root: {DATASET_ROOT}")
log.info(f"Log file: {LOG_FILE}")

# ==============================
# DETECTION CONFIG
# ==============================
DET_SIZE = (1024, 1024)
DET_THRESH = 0.25

MIN_FACE_INPUT = 256
MIN_FACE_SIZE = 64
MIN_FACE_SCORE = 0.35

# ==============================
# INIT INSIGHTFACE
# ==============================
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(
    ctx_id=-1,
    det_size=DET_SIZE,
    det_thresh=DET_THRESH
)

# ==============================
# HELPERS
# ==============================
def upscale_if_needed(img):
    h, w = img.shape[:2]
    if min(h, w) < MIN_FACE_INPUT:
        scale = MIN_FACE_INPUT / min(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        return img, True
    return img, False


def filter_faces(faces):
    valid = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        w = x2 - x1
        h = y2 - y1

        if f.det_score < MIN_FACE_SCORE:
            continue
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue

        valid.append(f)
    return valid

# ==============================
# MAIN
# ==============================
def main():
    identities = sorted(p for p in DATASET_ROOT.iterdir() if p.is_dir())
    log.info(f"Found {len(identities)} identities")

    total_faces = 0

    for person_dir in tqdm(identities, desc="People"):
        images = sorted(person_dir.glob("*.*"))
        log.info(f"[PERSON] {person_dir.name} | images={len(images)}")

        person_faces = 0

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning(f"Could not read {img_path.name}")
                continue

            shape = img.shape
            img, upscaled = upscale_if_needed(img)

            faces = app.get(img)
            valid_faces = filter_faces(faces)

            log.info(
                f"[IMG] {img_path.name} | "
                f"shape={shape} | "
                f"faces={len(valid_faces)} | "
                f"upscaled={upscaled}"
            )

            if faces and not valid_faces:
                log.warning("Faces detected but filtered out")
                for f in faces:
                    x1, y1, x2, y2 = f.bbox.astype(int)
                    log.warning(
                        f"RAW score={f.det_score:.3f} "
                        f"size=({x2-x1}x{y2-y1})"
                    )

            person_faces += len(valid_faces)

        if person_faces == 0:
            log.warning(f"Identity '{person_dir.name}' has only 0 usable faces")

        total_faces += person_faces

    log.info("===================================")
    log.info(f"DONE | Total faces collected: {total_faces}")
    log.info("===================================")

# ==============================
# ENTRY
# ==============================
if __name__ == "__main__":
    main()
