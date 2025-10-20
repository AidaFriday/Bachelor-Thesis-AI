import os
import cv2
import json
import time
import numpy as np
from tqdm import tqdm
import sys

# Add the "src" directory to Python path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from connector import load_model


def send_log(msg: str, level: str = "info"):
    """Emit a structured log line that the GUI will ignore (it only parses pure JSON objects)."""
    print(json.dumps({"log": msg, "level": level}))
    sys.stdout.flush()


def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings."""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ----------------- Deterministic pair builder (pos + neg) -----------------


def _collect_ordered_people_with_images(dataset_path):
    people = [
        p
        for p in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, p))
    ]
    people.sort()
    imgs_by_person = []
    for person in people:
        imgs = [
            os.path.join(dataset_path, person, f)
            for f in sorted(os.listdir(os.path.join(dataset_path, person)))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if imgs:
            imgs_by_person.append((person, imgs))
    return imgs_by_person  # list of (person, [img paths])


def _ring_index(i, n):  # helper for wrap-around
    return i % n if n else 0


def build_pairs_deterministic(
    dataset_path,
    start_person=None,
    max_pairs=600,
    pos_ratio=0.5,
    exclude_singletons=True,  # Option B: use only folders with ≥2 images for both pos & neg
):
    """
    Deterministic pair set (Option B):
      - Only use identities with >= 2 images for BOTH positives and negatives.
      - Start at `start_person` if available; if it is a singleton or missing,
        start at the next available >=2-image identity in alphabetical order.
      - Order is stable and reproducible.
    """
    pos_ratio = max(0.0, min(1.0, float(pos_ratio)))

    # 1) collect all people (alphabetical)
    people_all = _collect_ordered_people_with_images(dataset_path)
    if not people_all:
        print("[ERROR] No images found in dataset")
        return []

    # 2) filter to >=2 images if Option B
    if exclude_singletons:
        people = [(p, imgs) for (p, imgs) in people_all if len(imgs) >= 2]
    else:
        people = people_all[:]  # retained for completeness

    if not people:
        print("[ERROR] No identities with >=2 images were found; nothing to build")
        return []

    all_names = [p for p, _ in people_all]
    filt_names = [p for p, _ in people]

    # 3) find deterministic start index, honoring chosen person if possible
    if start_person:
        if start_person in filt_names:
            start_idx = filt_names.index(start_person)
        else:
            if start_person in all_names:
                base = all_names.index(start_person)
                chosen = None
                for k in range(len(all_names)):
                    name = all_names[(base + k) % len(all_names)]
                    if name in filt_names:
                        chosen = name
                        break
                if chosen is None:
                    start_idx = 0
                else:
                    start_idx = filt_names.index(chosen)
                    print(
                        f"[WARN] start_person '{start_person}' has <2 images; "
                        f"starting at next available '{chosen}'"
                    )
            else:
                print(
                    f"[WARN] start_person '{start_person}' not found; starting at '{filt_names[0]}'"
                )
                start_idx = 0
    else:
        start_idx = 0

    # 4) build positives (all combinations) from filtered people
    pos_pool = []
    for k in range(len(people)):  # walk from start_idx with wrap-around
        person_idx = _ring_index(start_idx + k, len(people))
        _, imgs = people[person_idx]
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                pos_pool.append((imgs[i], imgs[j], 1))

    # 5) build negatives (round-robin by index) from filtered people only
    neg_pool = []
    n = len(people)
    if n >= 2:
        for offset in range(1, n):
            for a in range(n):
                b = _ring_index(a + offset, n)
                imgs_a = people[a][1]
                imgs_b = people[b][1]
                L = min(len(imgs_a), len(imgs_b))
                for t in range(L):
                    neg_pool.append((imgs_a[t], imgs_b[t], 0))

    # 6) allocate counts + top-up deterministically
    want_pos = int(round(max_pairs * pos_ratio))
    want_neg = max_pairs - want_pos
    pos_take = pos_pool[:want_pos]
    neg_take = neg_pool[:want_neg]

    combined = pos_take + neg_take
    if len(combined) < max_pairs:
        rest_pos = pos_pool[len(pos_take) :]
        rest_neg = neg_pool[len(neg_take) :]
        for chunk in (rest_pos, rest_neg):
            for item in chunk:
                if len(combined) >= max_pairs:
                    break
                combined.append(item)

    print(
        f"[INFO] Built {len(combined)} pairs "
        f"({sum(1 for *_, l in combined if l==1)} pos / "
        f"{sum(1 for *_, l in combined if l==0)} neg) "
        f"starting at '{filt_names[start_idx]}'"
    )
    return combined[:max_pairs]


# ----------------- Eval utilities -----------------


def find_best_threshold(sims, labels):
    thresholds = np.linspace(-1, 1, 200)
    best_acc, best_t = 0.0, 0.0
    labels = np.array(labels)
    sims = np.array(sims)
    for t in thresholds:
        preds = (sims > t).astype(int)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc, best_t = float(acc), float(t)
    return best_acc, best_t


# ----------------- Main logic -----------------


def run_logic(
    model_path,
    iters=300,
    frame_h=None,
    frame_w=None,
    dataset_path=None,
    start_person=None,
    pos_ratio=0.5,
):
    if dataset_path is None:
        dataset_path = model_path

    # --- Auto-fix for common dataset folder structure ---
    if os.path.isdir(os.path.join(dataset_path, "lfw-deepfunneled")):
        dataset_path = os.path.join(dataset_path, "lfw-deepfunneled")

    wrapper = load_model(model_path)
    model_name = getattr(wrapper, "name", os.path.basename(model_path))

    # allow env overrides (useful when called from GUI)
    start_person = start_person or os.getenv("LFW_START_PERSON") or None
    try:
        pos_ratio = float(os.getenv("POS_RATIO", pos_ratio))
    except Exception:
        pass
    pos_ratio = max(0.0, min(1.0, pos_ratio))

    print(
        f"[DEBUG] run_logic() iters={iters}, dataset={dataset_path}, "
        f"start_person={start_person}, pos_ratio={pos_ratio}"
    )

    # --- Build pairs deterministically ---
    pairs = build_pairs_deterministic(
        dataset_path,
        start_person=start_person,  # ask via popup in GUI; or CLI flag
        max_pairs=iters,  # “How many pairs?” from popup/CLI
        pos_ratio=pos_ratio,  # 0.5 = balanced pos/neg
        exclude_singletons=True,  # Option B in effect
    )

    if not pairs:
        print(
            json.dumps(
                {
                    "kind": "accuracy_image",
                    "dataset": os.path.basename(dataset_path),
                    "model": model_name,
                    "num_pairs": 0,
                    "error": "No pairs could be built (check dataset path or start person)",
                }
            ),
            flush=True,
        )
        return

    # --- Evaluate pairs ---
    sims, labels = [], []
    start_time = time.time()

    for img1, img2, label in tqdm(pairs, desc="Validating", ncols=80):
        img1 = os.path.normpath(img1)
        img2 = os.path.normpath(img2)

        a = cv2.imread(img1)
        b = cv2.imread(img2)
        if a is None or b is None:
            print(f"[WARN] Skipping unreadable pair:\n  {img1}\n  {img2}")
            continue

        emb1 = wrapper.embed(a)
        emb2 = wrapper.embed(b)
        if emb1 is None or emb2 is None:
            print(f"[WARN] Skipping pair with missing embedding:\n  {img1}\n  {img2}")
            continue

        sims.append(cosine_similarity(emb1, emb2))
        labels.append(int(label))

    if not sims:
        print(
            json.dumps(
                {
                    "kind": "accuracy_image",
                    "dataset": os.path.basename(dataset_path),
                    "model": model_name,
                    "num_pairs": 0,
                    "error": "All pairs were unreadable or produced no embeddings",
                }
            ),
            flush=True,
        )
        return

    acc, best_t = find_best_threshold(sims, labels)
    elapsed = time.time() - start_time

    # --- Confusion counts at best threshold ---
    labels_np = np.array(labels, dtype=int)
    preds = (np.array(sims) > best_t).astype(int)

    tp = int(((preds == 1) & (labels_np == 1)).sum())
    tn = int(((preds == 0) & (labels_np == 0)).sum())
    fp = int(((preds == 1) & (labels_np == 0)).sum())
    fn = int(((preds == 0) & (labels_np == 1)).sum())

    pos = int((labels_np == 1).sum())
    neg = int((labels_np == 0).sum())

    # identities involved (folder names)
    def _identity_from_path(p):
        return os.path.basename(os.path.dirname(p))

    identities_used = sorted(
        set(
            [_identity_from_path(p1) for p1, _, _ in pairs]
            + [_identity_from_path(p2) for _, p2, _ in pairs]
        )
    )
    unique_identities = len(identities_used)
    identities_preview = identities_used[:8]

    result = {
        "kind": "accuracy_image",
        "dataset": os.path.basename(dataset_path),
        "model": model_name,
        "num_pairs": len(sims),  # effective evaluated pairs
        "requested_pairs": len(pairs),  # before skipping any invalids
        "pos_ratio": round(float(pos_ratio), 3),
        "accuracy": round(float(acc), 5),
        "threshold": round(float(best_t), 3),
        "elapsed_sec": round(float(elapsed), 2),
        "start_person": start_person,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "pos_pairs": pos,
        "neg_pairs": neg,
        "unique_identities": unique_identities,
        "identities_preview": identities_preview,
        "title": f"{model_name} – VA (Image) – Start: {start_person or 'N/A'}",
        "summary": f"TP:{tp} FP:{fp} TN:{tn} FN:{fn} | +:{pos} -:{neg} | IDs:{unique_identities}",
    }

    # Human-readable pretty block for the console
    print("[RESULT]", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    print("", flush=True)  # blank line for readability
    print(json.dumps(result), flush=True)


# ----------------- CLI -----------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model key for connector.load_model (e.g., arcface, facenet, insightface)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to LFW dataset or its parent (will auto-use lfw-deepfunneled)",
    )
    parser.add_argument(
        "--iters", type=int, default=300, help="Number of pairs to evaluate"
    )
    parser.add_argument(
        "--start-person",
        type=str,
        default=None,
        help="Folder/person name to start from (exact match)",
    )
    parser.add_argument(
        "--pos-ratio",
        type=float,
        default=0.5,
        help="Fraction of positives in final set (0.0..1.0). 0.5 = balanced",
    )
    args = parser.parse_args()

    run_logic(
        args.model_path,
        iters=args.iters,
        dataset_path=args.dataset_path,
        start_person=args.start_person,
        pos_ratio=args.pos_ratio,
    )
