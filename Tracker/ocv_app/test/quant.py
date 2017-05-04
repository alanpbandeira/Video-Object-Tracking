# from sklearn.cluster import MeanShift
from sklearn.cluster import MiniBatchKMeans

import numpy as np
import cv2

image = cv2.imread("Lenna.png")
h, w = image.shape[:2]

image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

image = image.reshape((h * w, 3))

clt = MiniBatchKMeans(8)
labels = clt.fit_predict(image)
print(labels)
quant = clt.cluster_centers_.astype("uint8")[labels]

quant = quant.reshape((h, w, 3))
image = image.reshape((h, w, 3))

quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

cv2.imshow('image', np.hstack([image, quant]))
# cv2.imshow('image', image)
cv2.waitKey(0)