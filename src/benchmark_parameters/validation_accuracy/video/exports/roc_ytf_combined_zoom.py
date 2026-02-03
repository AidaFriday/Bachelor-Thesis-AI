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
        "color": "#1f77b4",  # blue
    },
    "FaceNet": {
        "prefix": "facenet_ytf",
        "stamp": "20260105-150804",
        "color": "#2ca02c",  # green
    },
    "AdaFace": {
        "prefix": "adaface_ytf",
        "stamp": "20260105-120835",
        "color": "#d62728",  # red
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
        "eer": eer,
        "eer_fpr": fpr[eer_idx],
        "eer_tpr": tpr[eer_idx],
        "auc": roc_auc,
        "color": cfg["color"],
    }

# -----------------------------
# SLIDE-READY ZOOMED ROC PLOT
# -----------------------------
plt.figure(figsize=(8, 6))

for name, r in results.items():
    # ROC curve
    plt.plot(
        r["fpr"],
        r["tpr"],
        color=r["color"],
        linewidth=3,
        label=f"{name} (EER={r['eer']:.3f})",
    )

    # EER point (big & visible)
    plt.scatter(
        r["eer_fpr"],
        r["eer_tpr"],
        color=r["color"],
        s=140,
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )

    # EER guide lines (subtle)
    plt.axvline(r["eer_fpr"], color=r["color"], linestyle="--", alpha=0.25)
    plt.axhline(r["eer_tpr"], color=r["color"], linestyle="--", alpha=0.25)

# 🔍 SMART ZOOM around EER region
eer_fprs = [r["eer_fpr"] for r in results.values()]
eer_tprs = [r["eer_tpr"] for r in results.values()]

plt.xlim(max(0, min(eer_fprs) - 0.03), min(1, max(eer_fprs) + 0.04))
plt.ylim(max(0, min(eer_tprs) - 0.03), min(1, max(eer_tprs) + 0.02))

# Labels & style
plt.xlabel("False Positive Rate", fontsize=13)
plt.ylabel("True Positive Rate", fontsize=13)
plt.title("ROC Curves on YTF Dataset (Zoomed EER Region)", fontsize=15)

plt.legend(loc="lower right", fontsize=11, frameon=True)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("roc_ytf_zoom_presentation.png", dpi=300)
plt.show()
