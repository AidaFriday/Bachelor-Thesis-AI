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
    },
    "FaceNet": {
        "prefix": "facenet_ytf",
        "stamp": "20260105-150804",
    },
    "AdaFace": {
        "prefix": "adaface_ytf",
        "stamp": "20260105-120835",
    },
}


# -----------------------------
# LOAD ALL FOLDS
# -----------------------------
def load_all_folds(prefix, stamp):
    scores_all = []
    labels_all = []

    for fold in range(10):
        scores_path = FOLDS_DIR / f"{prefix}_fold{fold}_{stamp}_scores.npy"
        labels_path = FOLDS_DIR / f"{prefix}_fold{fold}_{stamp}_labels.npy"

        if not scores_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"Missing files for fold {fold}")

        scores_all.append(np.load(scores_path))
        labels_all.append(np.load(labels_path))

    scores = np.concatenate(scores_all)
    labels = np.concatenate(labels_all)

    return scores, labels


# -----------------------------
# PLOT ROC
# -----------------------------
plt.figure(figsize=(7, 6))

colors = {
    "ArcFace": "blue",
    "FaceNet": "green",
    "AdaFace": "red",
}

for name, cfg in MODELS.items():
    scores, labels = load_all_folds(cfg["prefix"], cfg["stamp"])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # --------- EER computation ---------
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    eer_fpr = fpr[eer_idx]
    eer_tpr = tpr[eer_idx]
    # ----------------------------------

    # ROC curve (color per model)
    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        color=colors[name],
        label=f"{name} (AUC={roc_auc:.3f}, EER={eer:.3f})",
    )

    # 🔴 EER point (same color as curve)
    plt.scatter(
        eer_fpr,
        eer_tpr,
        color=colors[name],
        s=80,
        marker="o",
        edgecolors="black",
        zorder=5,
    )


# random baseline
plt.plot([0, 1], [0, 1], "k--", linewidth=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves on YTF Dataset")
plt.legend(loc="lower right")
plt.grid(True)

# 🔍 ZOOM into EER region (ADD HERE)
plt.xlim(0, 0.12)
plt.ylim(0.90, 1.0)


plt.tight_layout()
plt.savefig("roc_ytf_all_models.png", dpi=300)
plt.show()
