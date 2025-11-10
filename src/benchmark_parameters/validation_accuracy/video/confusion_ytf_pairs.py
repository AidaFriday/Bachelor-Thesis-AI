import os
import sys
import json
from pathlib import Path
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

FIXED_THRESHOLD = 0.6  # or you can later plug in the ROC-optimal one


def load_scores_labels(base_prefix):
    """
    base_prefix is the path returned by export_ytf_pairs()
    without the _scores/_labels suffix.
    """
    base = Path(base_prefix)
    scores = np.load(base.with_name(base.name + "_scores.npy"))
    labels = np.load(base.with_name(base.name + "_labels.npy"))
    return scores, labels


def run_confusion_ytf(base_prefix, model_name="unknown"):
    scores, labels = load_scores_labels(base_prefix)

    preds = (scores >= FIXED_THRESHOLD).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    far = fp / (fp + tn + 1e-9)
    frr = fn / (fn + tp + 1e-9)

    result = {
        "kind": "confusion_matrix_ytf",
        "model": model_name,
        "dataset": "YTF",
        "pairs": int(len(labels)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "far": float(far),
        "frr": float(frr),
        "threshold": float(FIXED_THRESHOLD),
    }

    export_dir = Path(__file__).resolve().parents[2] / "exports"
    export_dir.mkdir(exist_ok=True, parents=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = export_dir / f"{model_name}_ytf_confusion_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[YTF CM] Exported -> {out_path}")
    print(json.dumps(result))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base prefix from logic_ytf_pairs")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    run_confusion_ytf(args.base, args.model)
