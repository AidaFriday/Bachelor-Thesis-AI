import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

np.random.seed(42)

# -------------------------------------------------
# Simulate realistic biometric scores
# genuine pairs -> high similarity
# impostor pairs -> low similarity
# -------------------------------------------------
n = 2000
genuine_scores = np.random.normal(loc=0.75, scale=0.08, size=n)
impostor_scores = np.random.normal(loc=0.35, scale=0.10, size=n)

scores = np.concatenate([genuine_scores, impostor_scores])
labels = np.concatenate([np.ones(n), np.zeros(n)])

# clip scores to [0,1]
scores = np.clip(scores, 0, 1)

# -------------------------------------------------
# ROC
# -------------------------------------------------
fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

# Youden's J
j_scores = tpr - fpr
youden_idx = np.argmax(j_scores)
youden_thr = thresholds[youden_idx]
youden_point = (fpr[youden_idx], tpr[youden_idx])

# EER
fnr = 1 - tpr
eer_idx = np.argmin(np.abs(fnr - fpr))
eer_thr = thresholds[eer_idx]
eer_point = (fpr[eer_idx], tpr[eer_idx])

# Manual thresholds (conceptual)
manual_thresholds = np.linspace(0.2, 0.8, 4)

# -------------------------------------------------
# Plot
# -------------------------------------------------
plt.figure(figsize=(7, 6))

# ROC curve
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")

# random baseline
plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random classifier")

# Youden point
plt.scatter(
    *youden_point, color="green", s=80, label=f"Youden's J (thr={youden_thr:.2f})"
)

# EER point
plt.scatter(*eer_point, color="red", s=80, label=f"EER (thr={eer_thr:.2f})")

# Manual thresholds (orange dots)
for thr in manual_thresholds:
    idx = np.argmin(np.abs(thresholds - thr))
    plt.scatter(fpr[idx], tpr[idx], color="orange", s=40)

plt.text(0.55, 0.25, "Manual thresholds\n(0.2 – 0.8)", color="orange")

# Labels
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Conceptual ROC Curve: Youden vs EER vs Manual Thresholds")
plt.legend(loc="lower right")
plt.grid(True)

plt.show()
