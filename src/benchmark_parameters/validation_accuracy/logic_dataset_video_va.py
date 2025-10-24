# ==== logic_dataset_video_va.py ====
import os, sys, json, time, numpy as np, cv2
from tqdm import tqdm

# Allow "src" imports when run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from connector import load_model


def send_log(msg: str, level: str = "info"):
    print(json.dumps({"log": msg, "level": level}))
    sys.stdout.flush()


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------- ROC helpers (same math as image VA) ----------
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


def _plot_and_save_roc(fpr, tpr, auc, eer, title, out_png, stats_box_text=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5.2, 4.6), facecolor="white")
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

    # keep a right gutter and draw box in figure coords
    fig.subplots_adjust(right=0.60)
    if stats_box_text:
        fig.text(
            0.965,
            0.84,
            stats_box_text,
            transform=fig.transFigure,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.75", alpha=0.95),
            clip_on=False,
            zorder=10,
        )
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------- Video dataset helpers ----------
def _collect_frames_by_identity_ytf(
    root_dir, max_frames_per_clip=5, extensions=(".jpg", ".jpeg", ".png")
):
    """
    YTF layout: person/clip_name/frame.jpg
    Return: list[(person, [frame_paths])]
    """
    people = []
    for person in sorted(os.listdir(root_dir)):
        pdir = os.path.join(root_dir, person)
        if not os.path.isdir(pdir):
            continue
        frames = []
        # walk clips under this person
        for clip in sorted(os.listdir(pdir)):
            cdir = os.path.join(pdir, clip)
            if not os.path.isdir(cdir):
                continue
            imgs = [
                os.path.join(cdir, f)
                for f in sorted(os.listdir(cdir))
                if f.lower().endswith(extensions)
            ]
            if imgs:
                frames.extend(imgs[:max_frames_per_clip])
        if frames:
            people.append((person, frames))
    return people


def _pair_key(a: str, b: str, lbl: int):
    a = os.path.normpath(a)
    b = os.path.normpath(b)
    return (a, b, int(lbl)) if a <= b else (b, a, int(lbl))


def _build_pairs_video(
    dataset_path,
    iters=600,
    pos_ratio=0.5,
    max_pos_per_identity=10,
    max_neg_per_identity=20,
    max_pos_per_image=3,
    max_neg_per_image=3,
):
    """
    Deterministic-ish pair builder over YTF-like tree.
    """
    # allow parent path -> aligned_images_DB
    if os.path.isdir(os.path.join(dataset_path, "aligned_images_DB")):
        dataset_path = os.path.join(dataset_path, "aligned_images_DB")

    people = _collect_frames_by_identity_ytf(dataset_path)
    if not people:
        return []

    pos_ratio = max(0.0, min(1.0, float(pos_ratio)))
    want_pos = int(round(iters * pos_ratio))
    want_neg = iters - want_pos

    # positives
    pos_pool, pos_seen = [], set()
    pos_count_id = {name: 0 for name, _ in people}
    pos_use_img = {}
    for name, frames in people:
        if pos_count_id[name] >= max_pos_per_identity:
            continue
        m = len(frames)
        added_for_id = 0
        for i in range(m):
            if added_for_id >= max_pos_per_identity:
                break
            for j in range(i + 1, m):
                if pos_use_img.get(frames[i], 0) >= max_pos_per_image:
                    continue
                if pos_use_img.get(frames[j], 0) >= max_pos_per_image:
                    continue
                key = _pair_key(frames[i], frames[j], 1)
                if key in pos_seen:
                    continue
                pos_seen.add(key)
                pos_pool.append((frames[i], frames[j], 1))
                pos_use_img[frames[i]] = pos_use_img.get(frames[i], 0) + 1
                pos_use_img[frames[j]] = pos_use_img.get(frames[j], 0) + 1
                pos_count_id[name] += 1
                added_for_id += 1
                if added_for_id >= max_pos_per_identity:
                    break

    # negatives
    neg_pool, neg_seen = [], set()
    neg_count_id = {name: 0 for name, _ in people}
    neg_use_img = {}
    n = len(people)
    for a in range(n):
        name_a, frames_a = people[a]
        for b in range(a + 1, n):
            name_b, frames_b = people[b]
            if (
                neg_count_id[name_a] >= max_neg_per_identity
                and neg_count_id[name_b] >= max_neg_per_identity
            ):
                continue
            L = min(len(frames_a), len(frames_b))
            for k in range(L):
                fa = frames_a[k % len(frames_a)]
                fb = frames_b[k % len(frames_b)]
                if neg_count_id[name_a] >= max_neg_per_identity:
                    break
                if neg_count_id[name_b] >= max_neg_per_identity:
                    break
                if neg_use_img.get(fa, 0) >= max_neg_per_image:
                    continue
                if neg_use_img.get(fb, 0) >= max_neg_per_image:
                    continue
                key = _pair_key(fa, fb, 0)
                if key in neg_seen:
                    continue
                neg_seen.add(key)
                neg_pool.append((fa, fb, 0))
                neg_use_img[fa] = neg_use_img.get(fa, 0) + 1
                neg_use_img[fb] = neg_use_img.get(fb, 0) + 1
                neg_count_id[name_a] += 1
                neg_count_id[name_b] += 1

    combined, seen = [], set()

    def _add(p):
        k = _pair_key(p[0], p[1], p[2])
        if k in seen:
            return False
        seen.add(k)
        combined.append(p)
        return True

    for p in pos_pool[:want_pos]:
        _add(p)
    for p in neg_pool[:want_neg]:
        _add(p)
    for pool in (pos_pool[want_pos:], neg_pool[want_neg:]):
        for p in pool:
            if len(combined) >= iters:
                break
            _add(p)
    return combined[:iters]


# ---------- Main entry (signature matches image VA call!) ----------
def run_logic(
    model_path,  # same param name as image logic
    iters=300,
    frame_h=None,
    frame_w=None,
    dataset_path=None,  # <-- IMPORTANT: accept dataset_path keyword
    start_person=None,
    pos_ratio=0.5,
):
    """
    Validation accuracy on *video* datasets (e.g., YTF).
    Collect frames per identity recursively and build pairs like image VA.
    Returns 'kind': 'accuracy_video' and also writes a ROC PNG+JSON so the GUI
    can render the full-bleed ROC image (already supported).
    """
    dataset_path = dataset_path or model_path

    # YTF convenience: if called on the parent, auto-use aligned_images_DB
    if os.path.isdir(os.path.join(dataset_path, "aligned_images_DB")):
        dataset_path = os.path.join(dataset_path, "aligned_images_DB")

    wrapper = load_model(model_path)
    model_name = getattr(wrapper, "name", os.path.basename(model_path))

    # build pairs
    pairs = _build_pairs_video(dataset_path, iters=iters, pos_ratio=pos_ratio)
    if not pairs:
        print(
            json.dumps(
                {
                    "kind": "accuracy_video",
                    "dataset": os.path.basename(dataset_path),
                    "model": model_name,
                    "num_pairs": 0,
                    "error": "No pairs could be built (check dataset path)",
                }
            ),
            flush=True,
        )
        return

    # evaluate
    sims, labels = [], []
    t0 = time.time()
    for a_path, b_path, lbl in tqdm(pairs, desc="Validating (video)", ncols=80):
        a = cv2.imread(a_path)
        b = cv2.imread(b_path)
        if a is None or b is None:
            continue
        ea = wrapper.embed(a)
        eb = wrapper.embed(b)
        if ea is None or eb is None:
            continue
        sims.append(_cosine(ea, eb))
        labels.append(int(lbl))
    if not sims:
        print(
            json.dumps(
                {
                    "kind": "accuracy_video",
                    "dataset": os.path.basename(dataset_path),
                    "model": model_name,
                    "num_pairs": 0,
                    "error": "All pairs unreadable or produced no embeddings",
                }
            ),
            flush=True,
        )
        return

    fixed_t = float(os.getenv("FIXED_THRESHOLD", "0.9"))
    labels_np = np.array(labels, dtype=int)
    preds = (np.array(sims) > fixed_t).astype(int)
    acc = float(np.mean(preds == labels_np))

    # confusion
    tp = int(((preds == 1) & (labels_np == 1)).sum())
    tn = int(((preds == 0) & (labels_np == 0)).sum())
    fp = int(((preds == 1) & (labels_np == 0)).sum())
    fn = int(((preds == 0) & (labels_np == 1)).sum())
    P = max(1, int((labels_np == 1).sum()))
    N = max(1, int((labels_np == 0).sum()))
    tpr_fixed = tp / P
    fpr_fixed = fp / N

    stats_box = (
        f"Threshold: {fixed_t:.3f}\n"
        f"TP: {tp}  FP: {fp}\n"
        f"TN: {tn}  FN: {fn}\n"
        f"TPR: {tpr_fixed:.3f}  FPR: {fpr_fixed:.3f}"
    )

    # ROC/AUC/EER
    fpr, tpr, thr = _roc_from_scores_labels(
        np.array(sims, dtype=np.float64), np.array(labels, dtype=np.int32)
    )
    auc = _auc_trapezoid(fpr, tpr)
    eer = _eer(fpr, tpr)

    elapsed = time.time() - t0

    # exports
    export_dir = os.path.join(os.path.dirname(__file__), "exports")
    os.makedirs(export_dir, exist_ok=True)
    dataset_name = os.path.basename(dataset_path.rstrip(os.sep))
    ts = time.strftime("%Y%m%d-%H%M%S")

    roc_png = os.path.join(export_dir, f"roc_{dataset_name}_{model_name}_{ts}.png")
    _plot_and_save_roc(
        fpr,
        tpr,
        auc,
        eer,
        f"ROC – {model_name} on {dataset_name}",
        roc_png,
        stats_box_text=stats_box,
    )

    roc_json = os.path.join(export_dir, f"roc_{dataset_name}_{model_name}_{ts}.json")
    with open(roc_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "dataset": dataset_name,
                "timestamp": ts,
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

    # GUI payload (matches windows/benchmark_window.py handlers)
    result = {
        "kind": "accuracy_video",
        "dataset": dataset_name,
        "model": model_name,
        "num_pairs": int(len(sims)),
        "accuracy": round(acc, 5),
        "threshold": round(fixed_t, 3),
        "elapsed_sec": round(float(elapsed), 2),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "auc": round(float(auc), 6),
        "eer": round(float(eer), 6),
        "roc_png": roc_png,
        "roc_json": roc_json,
    }
    # Print once pretty (for humans) and once compact (for GUI)
    print("[RESULT]", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    print(json.dumps(result), flush=True)
