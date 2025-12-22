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
        "stamp": "20251124-164339",
    },
    "FaceNet": {
        "prefix": "facenet_ytf",
        "stamp": "20251124-164242",
    },
    "AdaFace": {
        "prefix": "adaface_ytf",
        "stamp": "20251124-164152",
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

for name, cfg in MODELS.items():
    scores, labels = load_all_folds(cfg["prefix"], cfg["stamp"])

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC = {roc_auc:.3f})",
    )

# random baseline
plt.plot([0, 1], [0, 1], "k--", linewidth=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves on YTF Dataset")
plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout()
plt.savefig("roc_ytf_all_models.png", dpi=300)
plt.show()
