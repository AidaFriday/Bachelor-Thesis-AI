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


# ----------------- ROC helpers (pure additions) -----------------


def _roc_from_scores_labels(scores: np.ndarray, labels: np.ndarray, thresholds=None):
    """Return (fpr, tpr, thr) for descending thresholds."""
    if thresholds is None:
        thresholds = np.unique(scores)
    thresholds = np.sort(thresholds)[::-1]

    P = int(np.sum(labels == 1))
    N = int(np.sum(labels == 0))
    if P == 0 or N == 0:
        raise ValueError("Need both positive and negative pairs to compute ROC.")

    tpr_list, fpr_list, thr_list = [], [], []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        tpr = tp / P if P else 0.0
        fpr = fp / N if N else 0.0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        thr_list.append(float(t))

    # Ensure endpoints for a clean curve
    if fpr_list[-1] != 1.0 or tpr_list[-1] != 1.0:
        fpr_list.append(1.0)
        tpr_list.append(1.0)
        thr_list.append(thr_list[-1] - 1e-6)
    if fpr_list[0] != 0.0 or tpr_list[0] != 0.0:
        fpr_list.insert(0, 0.0)
        tpr_list.insert(0, 0.0)
        thr_list.insert(0, thr_list[0] + 1e-6)

    return np.asarray(fpr_list), np.asarray(tpr_list), np.asarray(thr_list)


def _auc_trapezoid(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def _eer(fpr: np.ndarray, tpr: np.ndarray) -> float:
    diff = np.abs(fpr - (1.0 - tpr))
    i = int(np.argmin(diff))
    if 0 < i < len(fpr):
        x1, y1 = fpr[i - 1], 1.0 - tpr[i - 1]
        x2, y2 = fpr[i], 1.0 - tpr[i]
        denom = (x2 - x1) - (y2 - y1)
        if abs(denom) > 1e-12:
            s = (y1 - x1) / denom
            s = min(max(s, 0.0), 1.0)
            x = x1 + s * (x2 - x1)
            return float(x)
    return float((fpr[i] + (1.0 - tpr[i])) / 2.0)


def _plot_and_save_roc(fpr, tpr, auc, eer, title, out_png, stats_box_text=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os  # <-- make sure os is available here

    fig = plt.figure(figsize=(5.2, 4.6))
    ax = fig.add_subplot(111)

    ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.scatter([eer], [1 - eer], s=28, zorder=5, label=f"EER ≈ {eer*100:.2f}%")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Reserve right gutter (smaller "right" => wider gutter)
    gutter_right = float(os.getenv("ROC_RIGHT_GUTTER", "0.60"))  # was 0.74 or 0.62
    fig.subplots_adjust(right=gutter_right)

    if stats_box_text:
        box_x = float(os.getenv("ROC_BOX_X", "0.965"))  # was 0.99
        box_y = float(os.getenv("ROC_BOX_Y", "0.84"))
        fig.text(
            box_x,
            box_y,
            stats_box_text,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.75", alpha=0.95),
            transform=fig.transFigure,
            zorder=10,
            clip_on=False,  # <- avoid clipping to containers
        )

    fig.savefig(out_png, dpi=150)
    plt.close(fig)


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
    exclude_singletons=True,  # only folders with ≥2 images for pos & neg
    max_pos_per_identity=10,  # cap positive pairs contributed by each identity
    max_neg_per_identity=20,  # cap negative pairs involving each identity
    max_pos_per_image=3,  # NEW: cap how many positive pairs each *image* can be in
    max_neg_per_image=3,  # NEW: cap how many negative pairs each *image* can be in
):
    """
    Deterministic (reproducible) pair set with:
      - Option B (no singletons)
      - per-identity caps
      - per-image caps (prevents one image from repeating a lot)
      - order-insensitive pair de-duplication
      - round-robin image selection to diversify pairs
    """
    pos_ratio = max(0.0, min(1.0, float(pos_ratio)))

    people_all = _collect_ordered_people_with_images(dataset_path)
    if not people_all:
        print("[ERROR] No images found in dataset")
        return []

    people = (
        [(p, imgs) for (p, imgs) in people_all if len(imgs) >= 2]
        if exclude_singletons
        else people_all[:]
    )
    if not people:
        print("[ERROR] No identities with >=2 images were found; nothing to build")
        return []

    all_names = [p for p, _ in people_all]
    filt_names = [p for p, _ in people]

    # start index resolution
    if start_person:
        if start_person in filt_names:
            start_idx = filt_names.index(start_person)
        else:
            if start_person in all_names:
                base = all_names.index(start_person)
                chosen = None
                for k in range(len(all_names)):
                    cand = all_names[(base + k) % len(all_names)]
                    if cand in filt_names:
                        chosen = cand
                        break
                start_idx = filt_names.index(chosen) if chosen else 0
                if chosen:
                    print(
                        f"[WARN] start_person '{start_person}' has <2 images; starting at next available '{chosen}'"
                    )
            else:
                print(
                    f"[WARN] start_person '{start_person}' not found; starting at '{filt_names[0]}'"
                )
                start_idx = 0
    else:
        start_idx = 0

    # --- POSITIVES (diversified & capped per image + per identity) ---
    pos_pool: list[tuple[str, str, int]] = []
    pos_seen: set[tuple[str, str, int]] = set()
    pos_count_id = {name: 0 for name, _ in people}
    pos_use_img: dict[str, int] = {}

    for k in range(len(people)):  # walk from start with wrap-around
        pi = _ring_index(start_idx + k, len(people))
        name, imgs = people[pi]
        if pos_count_id[name] >= max_pos_per_identity:
            continue

        added_for_id = 0
        m = len(imgs)
        for shift in range(1, m):
            for i in range(m):
                if added_for_id >= max_pos_per_identity:
                    break
                j = (i + shift) % m
                if i >= j:
                    continue
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

    # --- NEGATIVES (diversified & capped per image + per identity) ---
    neg_pool: list[tuple[str, str, int]] = []
    neg_seen: set[tuple[str, str, int]] = set()
    neg_count_id = {name: 0 for name, _ in people}
    neg_use_img: dict[str, int] = {}

    n = len(people)
    if n >= 2:
        for id_offset in range(1, n):
            for a_idx in range(n):
                b_idx = _ring_index(a_idx + id_offset, n)
                name_a, imgs_a = people[a_idx]
                name_b, imgs_b = people[b_idx]
                if (
                    neg_count_id[name_a] >= max_neg_per_identity
                    and neg_count_id[name_b] >= max_neg_per_identity
                ):
                    continue
                La, Lb = len(imgs_a), len(imgs_b)
                L = min(La, Lb)
                for img_offset in range(L):
                    a = imgs_a[img_offset % La]
                    b = imgs_b[(img_offset + id_offset) % Lb]
                    if neg_count_id[name_a] >= max_neg_per_identity:
                        continue
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

    want_pos = int(round(max_pairs * pos_ratio))
    want_neg = max_pairs - want_pos

    combined: list[tuple[str, str, int]] = []
    combined_seen: set[tuple[str, str, int]] = set()

    def _try_add(pair):
        key = _pair_key(pair[0], pair[1], pair[2])
        if key in combined_seen:
            return False
        combined.append(pair)
        combined_seen.add(key)
        return True

    for p in pos_pool[:want_pos]:
        _try_add(p)
    for q in neg_pool[:want_neg]:
        _try_add(q)

    if len(combined) < max_pairs:
        for pool in (pos_pool[want_pos:], neg_pool[want_neg:]):
            for item in pool:
                if len(combined) >= max_pairs:
                    break
                _try_add(item)

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

    # per-identity caps (allow env override if needed)
    try:
        max_pos_cap = int(os.getenv("MAX_POS_PER_ID", "10"))
    except Exception:
        max_pos_cap = 10
    try:
        max_neg_cap = int(os.getenv("MAX_NEG_PER_ID", "20"))
    except Exception:
        max_neg_cap = 20

    print(
        f"[DEBUG] run_logic() iters={iters}, dataset={dataset_path}, "
        f"start_person={start_person}, pos_ratio={pos_ratio}, "
        f"max_pos_per_identity={max_pos_cap}, max_neg_per_identity={max_neg_cap}"
    )

    # --- Build pairs deterministically ---
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

    # --- Evaluate pairs ---
    sims, labels = [], []
    start_time = time.time()

    total_pairs = len(pairs)
    # reset the bar to 0 first
    print(json.dumps({"_type": "progress", "progress": 0, "total": total_pairs}), flush=True)

    for i, (img1, img2, label) in enumerate(pairs):
        img1 = os.path.normpath(img1)
        img2 = os.path.normpath(img2)

        a = cv2.imread(img1)
        b = cv2.imread(img2)
        if a is None or b is None:
            print(f"[WARN] Skipping unreadable pair:\n  {img1}\n  {img2}")
            # still advance the visual progress so the bar moves
            print(json.dumps({"_type": "progress", "progress": i + 1, "total": total_pairs}), flush=True)
            continue

        emb1 = wrapper.embed(a)
        emb2 = wrapper.embed(b)
        if emb1 is None or emb2 is None:
            print(f"[WARN] Skipping pair with missing embedding:\n  {img1}\n  {img2}")
            print(json.dumps({"_type": "progress", "progress": i + 1, "total": total_pairs}), flush=True)
            continue

        sims.append(cosine_similarity(emb1, emb2))
        labels.append(int(label))

        # ✅ emit after every pair so the bar moves smoothly
        print(json.dumps({"_type": "progress", "progress": i + 1, "total": total_pairs}), flush=True)


    # emit progress updates periodically (every 10 pairs) and at the end
    if ((i + 1) % 10 == 0) or (i + 1 == total_pairs):
        print(
            json.dumps({"_type": "progress", "progress": i + 1, "total": total_pairs}),
            flush=True,
        )

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

    # --- Fixed-threshold accuracy (kept exactly as before) ---
    fixed_t = float(os.getenv("FIXED_THRESHOLD", "0.9"))  # default 0.9 if not set
    labels_np = np.array(labels, dtype=int)
    preds = (np.array(sims) > fixed_t).astype(int)
    acc = float(np.mean(preds == labels_np))
    best_t = fixed_t

    elapsed = time.time() - start_time

    # --- Confusion counts at fixed threshold ---
    labels_np = np.array(labels, dtype=int)
    preds = (np.array(sims) > best_t).astype(int)

    tp = int(((preds == 1) & (labels_np == 1)).sum())
    tn = int(((preds == 0) & (labels_np == 0)).sum())
    fp = int(((preds == 1) & (labels_np == 0)).sum())
    fn = int(((preds == 0) & (labels_np == 1)).sum())

    pos = int((labels_np == 1).sum())
    neg = int((labels_np == 0).sum())

    P = max(1, pos)  # positives
    N = max(1, neg)  # negatives
    tpr_at_fixed = tp / P
    fpr_at_fixed = fp / N
    stats_box_text = (
        f"Threshold: {best_t:.3f}\n"
        f"TP: {tp}  FP: {fp}\n"
        f"TN: {tn}  FN: {fn}\n"
        f"TPR: {tpr_at_fixed:.3f}  FPR: {fpr_at_fixed:.3f}"
    )

    # --- ROC/AUC/EER (new; exports PNG+JSON) ---
    fpr, tpr, thr = _roc_from_scores_labels(
        np.array(sims, dtype=np.float64), np.array(labels, dtype=np.int32)
    )
    auc = _auc_trapezoid(fpr, tpr)
    eer = _eer(fpr, tpr)

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
        "max_pos_per_identity": max_pos_cap,
        "max_neg_per_identity": max_neg_cap,
        "auc": round(float(auc), 6),
        "eer": round(float(eer), 6),
    }

    # ------- Export full run details to a separate JSON file -------
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    dataset_name = os.path.basename(dataset_path.rstrip(os.sep))
    test_name = "Validation Accuracy (Image)"

    # Store pairs with paths relative to dataset for portability
    pairs_export = []
    for a, b, lbl in pairs:
        pairs_export.append(
            {
                "a": os.path.relpath(os.path.normpath(a), dataset_path),
                "b": os.path.relpath(os.path.normpath(b), dataset_path),
                "label": "pos" if int(lbl) == 1 else "neg",
            }
        )

    export_dir = os.path.join(os.path.dirname(__file__), "exports")
    os.makedirs(export_dir, exist_ok=True)

    # Existing VA export (unchanged)
    export_path = os.path.join(
        export_dir, f"va_{dataset_name}_{model_name}_{timestamp}.json"
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
            # NEW:
            "auc": result["auc"],
            "eer": result["eer"],
        },
        "identities": identities_used,  # full, sorted list
        "pairs": pairs_export,  # every evaluated pair (no duplicates)
        # NEW: include raw ROC curve for reproducibility
        "roc_curve": {
            "fpr": [float(x) for x in fpr.tolist()],
            "tpr": [float(x) for x in tpr.tolist()],
            "thresholds": [float(x) for x in thr.tolist()],
        },
    }

    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    # NEW: Save ROC figure & a compact ROC-only JSON
    roc_png = os.path.join(
        export_dir, f"roc_{dataset_name}_{model_name}_{timestamp}.png"
    )
    _plot_and_save_roc(
        fpr,
        tpr,
        auc,
        eer,
        f"ROC – {model_name} on {dataset_name}",
        roc_png,
        stats_box_text=stats_box_text,  # <<< added
    )

    roc_json = os.path.join(
        export_dir, f"roc_{dataset_name}_{model_name}_{timestamp}.json"
    )
    with open(roc_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "dataset": dataset_name,
                "timestamp": timestamp,
                "auc": round(float(auc), 6),
                "eer": round(float(eer), 6),
                "figure_path": roc_png,
                "fpr": [float(x) for x in fpr.tolist()],
                "tpr": [float(x) for x in tpr.tolist()],
                "thresholds": [float(x) for x in thr.tolist()],
            },
            f,
            indent=2,
        )

    # Optional info for console
    result["export_path"] = export_path
    result["roc_png"] = roc_png  # NEW
    result["roc_json"] = roc_json  # NEW
    try:
        send_log(f"[export] wrote {export_path}")
        send_log(f"[export] wrote {roc_png}")
        send_log(f"[export] wrote {roc_json}")
    except NameError:
        pass

    # Human-readable pretty block for the console
    print("[RESULT]", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    print("", flush=True)  # blank line for readability

    # Raw JSON (single-line) for GUI to parse
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
