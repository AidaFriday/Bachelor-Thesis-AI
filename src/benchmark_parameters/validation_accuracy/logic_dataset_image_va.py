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


def _pair_key(a: str, b: str, lbl: int):
    """Order-insensitive unique key for a pair."""
    a = os.path.normpath(a)
    b = os.path.normpath(b)
    if a <= b:
        return (a, b, int(lbl))
    else:
        return (b, a, int(lbl))


def build_pairs_deterministic(
    dataset_path,
    start_person=None,
    max_pairs=600,
    pos_ratio=0.5,
    exclude_singletons=True,
    max_pos_per_identity=10,
    max_neg_per_identity=20,
    max_pos_per_image=3,
    max_neg_per_image=3,
):
    """
    Deterministic builder:
      • starts at start_person and goes FORWARD ONLY (NO WRAP)
      • skips identities with <2 images when exclude_singletons=True
      • caps positive/negative pairs per identity
      • builds negatives only among the forward slice
      • no randomness; stable across runs/models
    """
    pos_ratio = max(0.0, min(1.0, float(pos_ratio)))

    people_all = _collect_ordered_people_with_images(dataset_path)
    if not people_all:
        print("[ERROR] No images found in dataset")
        return []

    people = [
        (p, imgs)
        for (p, imgs) in people_all
        if (len(imgs) >= 2 or not exclude_singletons)
    ]
    if not people:
        print("[ERROR] No identities with >=2 images were found; nothing to build")
        return []

    # resolve start index inside the filtered list; forward-only (no wrap)
    names = [p for p, _ in people]
    if start_person and start_person in names:
        start_idx = names.index(start_person)
    else:
        if start_person:
            all_names = [p for p, _ in people_all]
            if start_person in all_names:
                # pick the first filtered identity alphabetically >= start_person
                start_idx = 0
                for i, (p, _) in enumerate(people):
                    if p >= start_person:
                        start_idx = i
                        break
            else:
                start_idx = 0
        else:
            start_idx = 0

    people_slice = people[start_idx:]
    if not people_slice:
        return []

    # ---------- POSITIVES ----------
    pos_pool, pos_seen = [], set()
    pos_count_id = {name: 0 for name, _ in people_slice}
    pos_use_img = {}

    for name, imgs in people_slice:
        if len(imgs) < 2 or pos_count_id[name] >= max_pos_per_identity:
            continue
        added_for_id = 0
        m = len(imgs)
        for shift in range(1, m):
            for i in range(m):
                if added_for_id >= max_pos_per_identity:
                    break
                j = i + shift
                if j >= m:
                    break  # no wrap within identity
                a, b = imgs[i], imgs[j]
                if pos_use_img.get(a, 0) >= max_pos_per_image:
                    continue
                if pos_use_img.get(b, 0) >= max_pos_per_image:
                    continue
                key = _pair_key(a, b, 1)
                if key in pos_seen:
                    continue
                pos_seen.add(key)
                pos_pool.append((a, b, 1))
                pos_use_img[a] = pos_use_img.get(a, 0) + 1
                pos_use_img[b] = pos_use_img.get(b, 0) + 1
                pos_count_id[name] += 1
                added_for_id += 1
            if added_for_id >= max_pos_per_identity:
                break

    # ---------- NEGATIVES ----------
    neg_pool, neg_seen = [], set()
    neg_count_id = {name: 0 for name, _ in people_slice}
    neg_use_img = {}

    n = len(people_slice)
    for a_idx in range(n):
        name_a, imgs_a = people_slice[a_idx]
        for b_idx in range(a_idx + 1, n):  # forward only, no wrap/back
            name_b, imgs_b = people_slice[b_idx]
            if (
                neg_count_id[name_a] >= max_neg_per_identity
                and neg_count_id[name_b] >= max_neg_per_identity
            ):
                continue
            La, Lb = len(imgs_a), len(imgs_b)
            L = min(La, Lb)
            for k in range(L):  # deterministic pairing
                a = imgs_a[k % La]
                b = imgs_b[k % Lb]
                if neg_count_id[name_a] >= max_neg_per_identity:
                    break
                if neg_count_id[name_b] >= max_neg_per_identity:
                    continue
                if neg_use_img.get(a, 0) >= max_neg_per_image:
                    continue
                if neg_use_img.get(b, 0) >= max_neg_per_image:
                    continue
                key = _pair_key(a, b, 0)
                if key in neg_seen:
                    continue
                neg_seen.add(key)
                neg_pool.append((a, b, 0))
                neg_use_img[a] = neg_use_img.get(a, 0) + 1
                neg_use_img[b] = neg_use_img.get(b, 0) + 1
                neg_count_id[name_a] += 1
                neg_count_id[name_b] += 1

    # ---------- combine to exactly max_pairs with wanted ratio ----------
    want_pos = int(round(max_pairs * pos_ratio))
    want_neg = max_pairs - want_pos

    combined, combined_seen = [], set()

    def _add(pair):
        k = _pair_key(pair[0], pair[1], pair[2])
        if k in combined_seen:
            return False
        combined.append(pair)
        combined_seen.add(k)
        return True

    for p in pos_pool:
        if sum(1 for *_, l in combined if l == 1) >= want_pos:
            break
        _add(p)

    for q in neg_pool:
        if sum(1 for *_, l in combined if l == 0) >= want_neg:
            break
        _add(q)

    # top up if one pool was short
    for pool in (pos_pool, neg_pool):
        for item in pool:
            if len(combined) >= max_pairs:
                break
            _add(item)

    return combined[:max_pairs]


# ----------------- Eval utilities -----------------


def _roc_curve(labels: np.ndarray, scores: np.ndarray):
    """
    Return FPR, TPR, thresholds for a binary classifier given positive=1 labels.
    No sklearn dependency.
    """
    order = np.argsort(-scores)
    y = labels[order].astype(int)
    s = scores[order]

    P = np.sum(y == 1)
    N = np.sum(y == 0)
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])

    tpr = [0.0]
    fpr = [0.0]
    thresholds = [np.inf]

    tp = 0
    fp = 0
    last_score = np.inf
    for yi, si in zip(y, s):
        if si != last_score:
            tpr.append(tp / P)
            fpr.append(fp / N)
            thresholds.append(si)
            last_score = si
        if yi == 1:
            tp += 1
        else:
            fp += 1

    tpr.append(tp / P)
    fpr.append(fp / N)
    thresholds.append(-np.inf)

    return np.asarray(fpr), np.asarray(tpr), np.asarray(thresholds)


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


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

    # Auto-fix for common dataset folder structure
    if os.path.isdir(os.path.join(dataset_path, "lfw-deepfunneled")):
        dataset_path = os.path.join(dataset_path, "lfw-deepfunneled")

    wrapper = load_model(model_path)
    model_name = getattr(wrapper, "name", os.path.basename(model_path))

    # env overrides (useful from GUI)
    start_person = start_person or os.getenv("LFW_START_PERSON") or None
    try:
        pos_ratio = float(os.getenv("POS_RATIO", pos_ratio))
    except Exception:
        pass
    pos_ratio = max(0.0, min(1.0, pos_ratio))

    try:
        max_pos_cap = int(os.getenv("MAX_POS_PER_ID", "10"))
    except Exception:
        max_pos_cap = 10
    try:
        max_neg_cap = int(os.getenv("MAX_NEG_PER_ID", "20"))
    except Exception:
        max_neg_cap = 20

    # Build pairs deterministically
    pairs = build_pairs_deterministic(
        dataset_path,
        start_person=start_person,
        max_pairs=iters,
        pos_ratio=pos_ratio,
        exclude_singletons=True,
        max_pos_per_identity=max_pos_cap,
        max_neg_per_identity=max_neg_cap,
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

    # Evaluate pairs
    sims, labels = [], []
    start_time = time.time()

    # progress -> STDERR so stdout stays clean
    for img1, img2, label in tqdm(pairs, desc="Validating", ncols=80, file=sys.stderr):
        img1 = os.path.normpath(img1)
        img2 = os.path.normpath(img2)

        a = cv2.imread(img1)
        b = cv2.imread(img2)
        if a is None or b is None:
            send_log(f"Skipping unreadable pair:\n  {img1}\n  {img2}", "warn")
            continue

        emb1 = wrapper.embed(a)
        emb2 = wrapper.embed(b)
        if emb1 is None or emb2 is None:
            send_log(
                f"Skipping pair with missing embedding:\n  {img1}\n  {img2}", "warn"
            )
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

    # fixed threshold (for apples-to-apples between models)
    fixed_t = float(os.getenv("FIXED_THRESHOLD", "0.7"))
    labels_np = np.array(labels, dtype=int)
    preds = (np.array(sims) > fixed_t).astype(int)
    acc = float(np.mean(preds == labels_np))
    best_t = fixed_t

    elapsed = time.time() - start_time

    # confusion counts
    tp = int(((preds == 1) & (labels_np == 1)).sum())
    tn = int(((preds == 0) & (labels_np == 0)).sum())
    fp = int(((preds == 1) & (labels_np == 0)).sum())
    fn = int(((preds == 0) & (labels_np == 1)).sum())
    pos = int((labels_np == 1).sum())
    neg = int((labels_np == 0).sum())

    # ROC + AUC (compute once)
    scores_np = np.array(sims, dtype=float)
    fpr, tpr, thr = _roc_curve(labels_np, scores_np)
    auc = _auc(fpr, tpr)

    # identities involved (order preserved from traversal)
    def _identity_from_path(p):
        return os.path.basename(os.path.dirname(p))

    identities_used = []
    _seen = set()
    for a, b, _ in pairs:
        for p in (a, b):
            name = _identity_from_path(p)
            if name not in _seen:
                _seen.add(name)
                identities_used.append(name)

    unique_identities = len(identities_used)
    identities_preview = identities_used[:8]

    result = {
        "kind": "accuracy_image",
        "dataset": os.path.basename(dataset_path),
        "model": model_name,
        "num_pairs": len(sims),
        "requested_pairs": len(pairs),
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
        "max_pos_per_identity": max_pos_cap,
        "max_neg_per_identity": max_neg_cap,
        "title": f"Model: {model_name} – Validation Accuracy (Image) – Start: {start_person or 'N/A'}",
        "summary": f"TP:{tp} FP:{fp} TN:{tn} FN:{fn} | +:{pos} -:{neg} | IDs:{unique_identities} | caps(+:{max_pos_cap}, -:{max_neg_cap})",
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": [float(x) for x in thr[:2000]],
            "auc": round(float(auc), 5),
        },
    }

    # ------- Export full run details to a separate JSON file -------
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    dataset_name = os.path.basename(dataset_path.rstrip(os.sep))
    test_name = "Validation Accuracy (Image)"

    pairs_export = []
    for a, b, lbl in pairs:
        pairs_export.append(
            {
                "a": os.path.relpath(os.path.normpath(a), dataset_path),
                "b": os.path.relpath(os.path.normpath(b), dataset_path),
                "label": "pos" if int(lbl) == 1 else "neg",
            }
        )

    export_payload = {
        "meta": {
            "model": model_name,
            "dataset": dataset_name,
            "test_name": test_name,
            "start_person": start_person,
            "pos_ratio": pos_ratio,
            "iters_requested": iters,
            "pairs_evaluated": len(sims),
            "timestamp": timestamp,
        },
        "stats": {
            "accuracy": result["accuracy"],
            "threshold": result["threshold"],
            "elapsed_sec": result["elapsed_sec"],
            "tp": result["tp"],
            "fp": result["fp"],
            "tn": result["tn"],
            "fn": result["fn"],
            "pos_pairs": result["pos_pairs"],
            "neg_pairs": result["neg_pairs"],
            "unique_identities": result["unique_identities"],
        },
        "identities": identities_used,  # order preserved
        "pairs": pairs_export,
    }

    export_dir = os.path.join(os.path.dirname(__file__), "exports")
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(
        export_dir, f"va_{dataset_name}_{model_name}_{timestamp}.json"
    )
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    result["export_path"] = export_path
    try:
        send_log(f"[export] wrote {export_path}")
    except NameError:
        pass

    # --- Human-friendly console markers (for the terminal only) ---
    print("[RESULT]")  # one-line marker, no pretty dump
    print(f"[SUMMARY] {result['summary']}")

    # FINAL single-line JSON for GUI to parse
    print(json.dumps(result), flush=True)


# ----------------- CLI -----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # accept both long forms for robustness
    parser.add_argument("--model_path", "--model", dest="model_path", required=True)
    parser.add_argument(
        "--dataset_path", "--dataset", dest="dataset_path", required=True
    )
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--start-person", type=str, default=None)
    parser.add_argument("--pos-ratio", type=float, default=0.5)
    args = parser.parse_args()

    run_logic(
        args.model_path,
        iters=args.iters,
        dataset_path=args.dataset_path,
        start_person=args.start_person,
        pos_ratio=args.pos_ratio,
    )
