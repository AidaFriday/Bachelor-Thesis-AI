# plot_roc_custom_all_models.py
import json
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(__file__)

files = {
    "ArcFace": "arcface_customroc_20260103-161342.json",
    "FaceNet": "facenet_original_customroc_20260103-161139.json",
    "AdaFace": "adaface_original_customroc_20260103-162313.json",
}

plt.figure(figsize=(7, 6))

for display_name, fname in files.items():
    path = os.path.join(BASE_DIR, fname)

    with open(path, "r") as f:
        data = json.load(f)

    # pooled ROC (already computed in your pipeline)
    fpr = data["fpr"]
    tpr = data["tpr"]
    auc = data["auc"]

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{display_name} (AUC = {auc:.3f})",
    )

# random baseline
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves on Custom Image-Based Dataset")
plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout()
plt.savefig("roc_custom_all_models.png", dpi=300)
plt.show()
