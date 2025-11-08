import os
import sys
import json
from datetime import datetime

import cv2
import numpy as np

# make project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from connector import load_model


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def dataset_needs_alignment(dataset_path: str) -> bool:
    """
    Return False for already-aligned datasets (e.g. LFW-deepfunneled),
    True for "raw" datasets.
    """
    path = dataset_path.lower()
    if "lfw" in path and "deepfunneled" in path:
        return False
    return True


def load_lfw_pairs_with_folds(pairs_file, dataset_path):
    """
    Same loader as in roc_lfw_pairs.py – builds the 6000 pairs for LFW.
    """
    pairs = []
    fold_ids = []
    with open(pairs_file, "r") as f:
        lines = f.read().strip().split("\n")

    num_folds, pairs_per_fold = map(int, lines[0].split())
    idx = 1

    for fold in range(num_folds):  # 0..9
        # positive pairs
        for _ in range(pairs_per_fold):
            name, n1, n2 = lines[idx].split()
            img1 = os.path.join(dataset_path, name, f"{name}_{int(n1):04d}.jpg")
            img2 = os.path.join(dataset_path, name, f"{name}_{int(n2):04d}.jpg")
            pairs.append((img1, img2, 1))
            fold_ids.append(fold)
            idx += 1

        # negative pairs
        for _ in range(pairs_per_fold):
            name1, n1, name2, n2 = lines[idx].split()
            img1 = os.path.join(dataset_path, name1, f"{name1}_{int(n1):04d}.jpg")
            img2 = os.path.join(dataset_path, name2, f"{name2}_{int(n2):04d}.jpg")
            pairs.append((img1, img2, 0))
            fold_ids.append(fold)
            idx += 1

    return pairs, np.array(fold_ids, dtype=np.int32)


def find_best_global_threshold(sims: np.ndarray, labels: np.ndarray):
    """
    Scan all unique similarity scores and pick the threshold
    that maximizes overall accuracy.
    """
    cand_thr = np.unique(sims)
    best_thr = None
    best_acc = -1.0

    for t in cand_thr:
        preds = (sims >= t).astype(np.int32)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc = acc
            best_thr = t

    return float(best_thr), float(best_acc)


def run_confusion_protocol(model_name, dataset_path, pairs_file):
    # if user passed parent LFW folder, go into lfw-deepfunneled
    if os.path.isdir(os.path.join(dataset_path, "lfw-deepfunneled")):
        dataset_path = os.path.join(dataset_path, "lfw-deepfunneled")

    print(f"[CM] Model   : {model_name}")
    print(f"[CM] Dataset : {dataset_path}")
    print(f"[CM] Pairs   : {pairs_file}")

    wrapper = load_model(model_name)
    model_name_lower = getattr(wrapper, "name", "").lower()
    is_arcface = model_name_lower == "arcface"
    is_adaface = model_name_lower == "adaface"

    pairs, fold_ids = load_lfw_pairs_with_folds(pairs_file, dataset_path)

    sims = []
    labels = []

    total = len(pairs)
    use_detection = dataset_needs_alignment(dataset_path)

    # only non-ArcFace, non-AdaFace models may use embed_aligned
    use_aligned = (
        (not use_detection)
        and (not is_arcface)
        and (not is_adaface)
        and hasattr(wrapper, "embed_aligned")
    )

    for i, (img1, img2, label) in enumerate(pairs, start=1):
        if i == 1 or i % 200 == 0 or i == total:
            print(f"[CM] Pair {i}/{total} ({(i/total)*100:.1f}%)")

        a = cv2.imread(img1)
        b = cv2.imread(img2)

        error = False
        emb1 = emb2 = None

        if a is None or b is None:
            error = True
        else:
            try:
                if is_arcface:
                    # ArcFace: use its tested path (internal detection/alignment)
                    emb1 = wrapper.get_embedding(img1)
                    emb2 = wrapper.get_embedding(img2)

                elif is_adaface:
                    # AdaFace: use its own embed() which expects an aligned crop
                    emb1 = wrapper.embed(a)
                    emb2 = wrapper.embed(b)

                elif use_aligned:
                    # Other models that have embed_aligned on aligned datasets
                    emb1 = wrapper.embed_aligned(a)
                    emb2 = wrapper.embed_aligned(b)

                elif use_detection:
                    # Facenet / others on non-aligned datasets
                    faces_a = wrapper.detector.detect(a)
                    faces_b = wrapper.detector.detect(b)
                    if not faces_a or not faces_b:
                        error = True
                    else:
                        aligned_a = wrapper.detector.align_for(a, faces_a[0]["kps"])
                        aligned_b = wrapper.detector.align_for(b, faces_b[0]["kps"])
                        if aligned_a is None or aligned_b is None:
                            error = True
                        else:
                            emb1 = wrapper.embed(aligned_a)
                            emb2 = wrapper.embed(aligned_b)
                else:
                    # Already aligned dataset, generic embed
                    emb1 = wrapper.embed(a)
                    emb2 = wrapper.embed(b)

            except Exception:
                error = True

        if emb1 is None or emb2 is None:
            error = True

        if error:
            # worst-case scores so they are misclassified for any thr in [0,1]
            sim = -1.0 if label == 1 else 2.0
        else:
            sim = cosine_similarity(emb1, emb2)

        sims.append(sim)
        labels.append(label)

    sims = np.array(sims, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # ---------- choose best global threshold ----------
    best_thr, best_acc = find_best_global_threshold(sims, labels)

    print(f"\n[CM] Best global threshold: {best_thr:.4f}")
    print(f"[CM] Accuracy at best thr: {best_acc*100:.2f}%")

    # ---------- confusion matrix at that threshold ----------
    preds = (sims >= best_thr).astype(np.int32)
    labels_np = np.asarray(labels, dtype=np.int32)

    # counts
    tp = int(np.sum((preds == 1) & (labels_np == 1)))
    tn = int(np.sum((preds == 0) & (labels_np == 0)))
    fp = int(np.sum((preds == 1) & (labels_np == 0)))
    fn = int(np.sum((preds == 0) & (labels_np == 1)))

    total_pairs = tp + tn + fp + fn

    # core metrics
    accuracy = (tp + tn) / total_pairs if total_pairs > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # specificity
    fpr = 1.0 - tnr
    fnr = 1.0 - recall

    total_pairs = len(labels)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall / sensitivity
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # specificity
    fpr = 1.0 - tnr
    fnr = 1.0 - tpr

    print("\n[CM] Confusion Matrix (at best threshold)")
    print("        Predicted 0   Predicted 1")
    print(f"Actual 0    TN={tn:4d}       FP={fp:4d}")
    print(f"Actual 1    FN={fn:4d}       TP={tp:4d}\n")

    print(f"[CM] Accuracy  : {accuracy*100:.2f}%")
    print(f"[CM] Precision : {precision*100:.2f}%")
    print(f"[CM] Recall    : {recall*100:.2f}%")
    print(f"[CM] F1-score  : {f1*100:.2f}%")
    print(f"[CM] TNR (specificity) : {tnr*100:.2f}%")
    print(f"[CM] FPR : {fpr*100:.2f}%")
    print(f"[CM] FNR : {fnr*100:.2f}%")

    # ---------- optional: plot and save confusion matrix ----------
    try:
        import matplotlib.pyplot as plt

        cm = np.array([[tn, fp], [fn, tp]], dtype=np.int32)
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)

        classes = ["Negative", "Positive"]
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes,
            yticklabels=classes,
            ylabel="Actual label",
            xlabel="Predicted label",
            title=f"Confusion Matrix – {model_name}",
        )

        # write numbers in cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        fig.tight_layout()

        exports_dir = os.path.join(os.path.dirname(__file__), "exports")
        os.makedirs(exports_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        png_path = os.path.join(exports_dir, f"{model_name}_confusion_{ts}.png")
        plt.savefig(png_path, dpi=150)
        plt.close(fig)

        print(f"[CM] Saved confusion matrix PNG to: {png_path}")
    except Exception as e:
        print(f"[CM] Warning: could not save confusion matrix figure ({e})")

    # ---------- JSON summary ----------
    result = {
        "kind": "confusion_matrix",
        "model": model_name,
        "dataset": os.path.basename(dataset_path),
        "pairs": int(total_pairs),
        "threshold": float(best_thr),
        "accuracy": float(best_acc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }

    print("\n[CM] JSON summary:")
    print(json.dumps(result, indent=2))

    # save JSON next to ROC exports
    exports_dir = os.path.join(os.path.dirname(__file__), "exports")
    os.makedirs(exports_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(exports_dir, f"{model_name}_confusion_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[CM] Saved confusion JSON to: {json_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pairs", type=str, required=True)
    args = parser.parse_args()

    run_confusion_protocol(args.model, args.dataset, args.pairs)
