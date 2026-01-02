import cv2
import numpy as np
from wrap_adaface_original import AdaFaceOriginalWrapper

# Load model
model = AdaFaceOriginalWrapper(device="cpu")

# Load LFW image
img_path = "/home/aida/Datasets/LFW/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
img = cv2.imread(img_path)

print("Image loaded:", img is not None)
print("Image shape:", img.shape if img is not None else None)

# Compute embedding
emb = model.embed(img)

print("Embedding type:", type(emb))
print("Embedding shape:", emb.shape)
print("Embedding dtype:", emb.dtype)
print("Embedding norm:", np.linalg.norm(emb))
