import numpy as np, json, os

exp = r"C:\programming\Bachelor-Thesis-AI\src\benchmark_parameters\exports"
scores = np.load(os.path.join(exp, "facenet_ytf_fold0_20251110-182757_scores.npy"))
labels = np.load(os.path.join(exp, "facenet_ytf_fold0_20251110-182757_labels.npy"))
print(scores.shape, labels.shape)   # (500,), (500,)
print("pos:", (labels==1).sum(), "neg:", (labels==0).sum())  # expect 250 / 250
