import json
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(__file__)

files = {
    "ArcFace": "arcface_roc_20251124-180735.json",
    "FaceNet": "facenet_original_roc_20251124-152518.json",
    "AdaFace": "adaface_roc_20251124-164558.json",
}

plt.figure(figsize=(7, 6))

for display_name, fname in files.items():
    path = os.path.join(BASE_DIR, fname)

    with open(path, "r") as f:
        data = json.load(f)

    fpr = data["roc_fpr"]
    tpr = data["roc_tpr"]
    auc = data["auc"]

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{display_name} (AUC = {auc:.3f})",
    )

# random baseline
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves on LFW Dataset")
plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout()
plt.savefig("roc_lfw_all_models.png", dpi=300)
plt.show()
