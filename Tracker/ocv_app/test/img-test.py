import img_processing as ipro
import cv2
import numpy as np


def test(a, b):
    print (a, b)
    # return [x + y for x, y in zip(a, b)]
    return a + b

a = [np.array((5,2)), np.array((2,1))]
b = np.array((5,2))

vfunc = np.vectorize(test, excluded=['a','b'])
print(vfunc(a,b))