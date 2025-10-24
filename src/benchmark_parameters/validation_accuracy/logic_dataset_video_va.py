import json, sys, time, numpy as np, os, cv2
from connector import load_model

try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def send_log(msg, level="info"):
    print(json.dumps({"log": msg, "level": level}))
    sys.stdout.flush()


def _cuda_sync():
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


# ----------------- ROC helpers (copy from image VA) -----------------


def _roc_from_scores_labels(scores: np.ndarray, labels: np.ndarray, thresholds=None):
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
        tpr_list.append(tp / P if P else 0.0)
        fpr_list.append(fp / N if N else 0.0)
        thr_list.append(float(t))
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


def _plot_and_save_roc(fpr, tpr, auc, eer, title, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5.2, 4.6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.scatter([eer], [1 - eer], s=28, zorder=5, label=f"EER ≈ {eer*100:.2f}%")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ----------------- Pair building for YTF-like directory -----------------


def _gather_images_by_identity(root_dir):
    """Return list of (identity_name, [image_paths]) under aligned_images_DB or given root."""
    aligned = os.path.join(root_dir, "aligned_images_DB")
    if os.path.exists(aligned):
        root_dir = aligned
    people = [
        d
        for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    groups = []
    for person in people:
        imgs = []
        person_dir = os.path.join(root_dir, person)
        for r, _, files in os.walk(person_dir):
            for f in files:
                if f.lower().endswith(".jpg"):
                    imgs.append(os.path.join(r, f))
        if imgs:
            imgs.sort()
            groups.append((person, imgs))
    return groups, root_dir


def _build_pairs_video(root_dir, max_pairs=600, pos_ratio=0.5, selected_subjects=None):
    """Deterministic positive/negative pairs from video frame folders."""
    groups, root_dir = _gather_images_by_identity(root_dir)
    if selected_subjects:
        selected = set(s.strip().lower() for s in selected_subjects)
        groups = [(p, imgs) for (p, imgs) in groups if p.lower() in selected]
    if not groups:
        return []

    # Filter identities with >=2 images for positives
    pos_groups = [(p, imgs) for (p, imgs) in groups if len(imgs) >= 2]
    if not pos_groups:
        return []

    want_pos = int(round(max_pairs * max(0.0, min(1.0, float(pos_ratio)))))
    want_neg = max_pairs - want_pos

    # Positives: round-robin inside each identity (diversified)
    pos_pairs = []
    for p, imgs in pos_groups:
        m = len(imgs)
        for shift in range(1, m):
            for i in range(m):
                j = (i + shift) % m
                if i >= j:
                    continue
                pos_pairs.append((imgs[i], imgs[j], 1))
                if len(pos_pairs) >= want_pos:
                    break
            if len(pos_pairs) >= want_pos:
                break
        if len(pos_pairs) >= want_pos:
            break

    # Negatives: pair across identities with offsets
    neg_pairs = []
    n = len(groups)
    for id_offset in range(1, n):
        for a_idx in range(n):
            b_idx = (a_idx + id_offset) % n
            pa, ia = groups[a_idx]
            pb, ib = groups[b_idx]
            L = min(len(ia), len(ib))
            for k in range(L):
                a = ia[k]
                b = ib[(k + id_offset) % len(ib)]
                neg_pairs.append((a, b, 0))
                if len(neg_pairs) >= want_neg:
                    break
            if len(neg_pairs) >= want_neg:
                break
        if len(neg_pairs) >= want_neg:
            break

    combined = pos_pairs + neg_pairs
    if len(combined) < max_pairs:
        # top up from remaining pools if needed
        for pool in (pos_pairs[want_pos:], neg_pairs[want_neg:]):
            for item in pool:
                combined.append(item)
                if len(combined) >= max_pairs:
                    break
            if len(combined) >= max_pairs:
                break

    return combined[:max_pairs]


def run_logic(model_name, iters, frame_h, frame_w, dataset):
    wrapper = load_model(model_name)

    # Subjects from GUI dialog (optional)
    selected_env = os.getenv("YTF_SELECTED_SUBJECTS", "")
    selected_subjects = [s.strip() for s in selected_env.split(",") if s.strip()]
    pos_ratio = float(os.getenv("POS_RATIO", "0.5"))

    send_log(
        f"Subjects: {', '.join(selected_subjects) if selected_subjects else 'all'}"
    )

    # Build deterministic pairs from video frames
    pairs = _build_pairs_video(
        dataset,
        max_pairs=iters,
        pos_ratio=pos_ratio,
        selected_subjects=selected_subjects,
    )

    if not pairs:
        print(
            json.dumps(
                {
                    "source_file": os.path.basename(__file__),
                    "kind": "accuracy_video",
                    "dataset": dataset,
                    "num_runs": int(os.getenv("YTF_RUNS", "1")),
                    "model": model_name,
                    "num_pairs": 0,
                    "error": "No pairs could be built (check dataset path/subjects).",
                }
            )
        )
        sys.stdout.flush()
        return

    # Warmup (unchanged logic spirit)
    first_img = cv2.imread(pairs[0][0])
    if first_img is None:
        first_img = np.random.randint(
            0, 255, (frame_h or 160, frame_w or 160, 3), np.uint8
        )
    warmup_iters = 5 if (_HAS_TORCH and torch.cuda.is_available()) else 1
    device_name = "GPU" if (_HAS_TORCH and torch.cuda.is_available()) else "CPU"
    send_log(f"🔥 Performing {warmup_iters} warm-up iteration(s) on {device_name}")
    for _ in range(warmup_iters):
        _ = wrapper.embed(first_img)
        _cuda_sync()

    # Evaluate similarities
    sims, labels = [], []
    start_time = time.time()
    for i, (p1, p2, lbl) in enumerate(pairs, 1):
        a = cv2.imread(p1)
        b = cv2.imread(p2)
        if a is None or b is None:
            continue
        ea = wrapper.embed(a)
        eb = wrapper.embed(b)
        if ea is None or eb is None:
            continue
        sims.append(float(np.dot(ea / np.linalg.norm(ea), eb / np.linalg.norm(eb))))
        labels.append(int(lbl))
        if (i % 25) == 0 or i == len(pairs):
            print(
                json.dumps({"_type": "progress", "progress": i, "total": len(pairs)}),
                flush=True,
            )

    if not sims:
        print(
            json.dumps(
                {
                    "source_file": os.path.basename(__file__),
                    "kind": "accuracy_video",
                    "dataset": dataset,
                    "num_runs": int(os.getenv("YTF_RUNS", "1")),
                    "model": model_name,
                    "num_pairs": 0,
                    "error": "All pairs unreadable or embeddings missing.",
                }
            )
        )
        sys.stdout.flush()
        return

    scores = np.array(sims, dtype=np.float64)
    lbls = np.array(labels, dtype=np.int32)

    # ROC metrics
    fpr, tpr, thr = _roc_from_scores_labels(scores, lbls)
    auc = _auc_trapezoid(fpr, tpr)
    eer = _eer(fpr, tpr)
    elapsed = time.time() - start_time

    # Fixed threshold accuracy (keep parity with image flow)
    fixed_t = float(os.getenv("FIXED_THRESHOLD", "0.9"))
    preds = (scores > fixed_t).astype(int)
    acc = float(np.mean(preds == lbls))

    # Exports
    export_dir = os.path.join(os.path.dirname(__file__), "exports")
    os.makedirs(export_dir, exist_ok=True)
    dataset_name = os.path.basename((dataset or "").rstrip(os.sep))
    if os.path.isdir(os.path.join(dataset, "aligned_images_DB")):
        dataset_name = "YTF_aligned"
    stamp = time.strftime("%Y%m%d-%H%M%S")

    roc_png = os.path.join(export_dir, f"roc_{dataset_name}_{model_name}_{stamp}.png")
    _plot_and_save_roc(
        fpr, tpr, auc, eer, f"ROC – {model_name} on {dataset_name}", roc_png
    )

    roc_json = os.path.join(export_dir, f"roc_{dataset_name}_{model_name}_{stamp}.json")
    with open(roc_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "dataset": dataset_name,
                "timestamp": stamp,
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

    payload = {
        "source_file": os.path.basename(__file__),
        "kind": "accuracy_video",
        "dataset": dataset,
        "num_runs": int(os.getenv("YTF_RUNS", "1")),
        "model": model_name,
        "subjects": selected_subjects,
        "num_pairs": int(len(scores)),
        "accuracy": acc,  # fixed-threshold accuracy (for parity)
        "auc": round(float(auc), 6),  # NEW
        "eer": round(float(eer), 6),  # NEW
        "elapsed_sec": round(float(elapsed), 2),
        "roc_png": roc_png,  # NEW
        "roc_json": roc_json,  # NEW
    }

    print(json.dumps(payload))
    sys.stdout.flush()
