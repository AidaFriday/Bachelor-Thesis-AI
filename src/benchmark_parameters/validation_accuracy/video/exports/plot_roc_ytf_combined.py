import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).parent
FOLDS_DIR = BASE_DIR / "folds"

MODELS = {
    "ArcFace": {
        "prefix": "arcface_ytf",
        "stamp": "20260105-155449",
        "color": "blue",
    },
    "FaceNet": {
        "prefix": "facenet_ytf",
        "stamp": "20260105-150804",
        "color": "green",
    },
    "AdaFace": {
        "prefix": "adaface_ytf",
        "stamp": "20260105-120835",
        "color": "red",
    },
}


# -----------------------------
# LOAD ALL FOLDS
# -----------------------------
def load_all_folds(prefix, stamp):
    scores_all, labels_all = [], []

    for fold in range(10):
        scores_path = FOLDS_DIR / f"{prefix}_fold{fold}_{stamp}_scores.npy"
        labels_path = FOLDS_DIR / f"{prefix}_fold{fold}_{stamp}_labels.npy"

        scores_all.append(np.load(scores_path))
        labels_all.append(np.load(labels_path))

    return np.concatenate(scores_all), np.concatenate(labels_all)


# -----------------------------
# COMPUTE ROC + EER
# -----------------------------
results = {}

for name, cfg in MODELS.items():
    scores, labels = load_all_folds(cfg["prefix"], cfg["stamp"])

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0

    results[name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
        "eer": eer,
        "eer_fpr": fpr[eer_idx],
        "eer_tpr": tpr[eer_idx],
        "color": cfg["color"],
    }

# -----------------------------
# SLIDE 1: FULL ROC PLOT
# -----------------------------
plt.figure(figsize=(7.5, 6.5))

for name, r in results.items():
    plt.plot(
        r["fpr"],
        r["tpr"],
        color=r["color"],
        linewidth=2.5,
        label=f"{name} (AUC={r['auc']:.3f}, EER={r['eer']:.3f})",
    )

    # EER point
    plt.scatter(r["eer_fpr"], r["eer_tpr"], color=r["color"], s=70, zorder=5)

# baseline
plt.plot([0, 1], [0, 1], "k--", linewidth=1)

plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves on YTF Dataset", fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("roc_ytf_full.png", dpi=300)
plt.show()
