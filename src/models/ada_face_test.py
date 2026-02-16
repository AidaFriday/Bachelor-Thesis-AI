import cv2
import numpy as np
from wrap_adaface_original import AdaFaceOriginalWrapper

model = AdaFaceOriginalWrapper(device="cpu")

img1 = cv2.imread(
    "/home/aida/Datasets/LFW/lfw-deepfunneled/George_W_Bush/George_W_Bush_0001.jpg"
)
img2 = cv2.imread(
    "/home/aida/Datasets/LFW/lfw-deepfunneled/George_W_Bush/George_W_Bush_0002.jpg"
)

assert img1 is not None
assert img2 is not None

e1 = model.embed(img1)
e2 = model.embed(img2)

print("same identity cosine:", float(np.dot(e1, e2)))
