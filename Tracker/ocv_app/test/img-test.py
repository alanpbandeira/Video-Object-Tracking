import img_processing as ipro
import cv2
import numpy as np

a = [(0.0, 0.0, 49.0)] * 600
# a = [(250.0, 2.0, 108.0)] * 600
a = np.array(a)

# cv2.imshow('window', a.reshape(20, 30, 3))
# cv2.waitKey(0)

print(a.reshape(20, 30, 3))