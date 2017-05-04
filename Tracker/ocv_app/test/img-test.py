import img_processing as ipro
import cv2
import numpy as np

def set_rgb_avarage(img):
    """docstring"""

    # rgb_data = []

    # for y in range(self.bitmask_map.shape[0]):
    #     for x in range(self.bitmask_map.shape[1]):
    #         if np.array_equal(
    #             self.bitmask_map[y,x],  np.array([1.0, 1.0, 1.0])):
    #             rgb_data.append(self.obj_features[x*y])
    
    # self.rgb_avarage = sum(rgb_data) / len(rgb_data)
    

    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]

    r = np.mean(r)
    g = np.mean(g)
    b = np.mean(b)
    
    return np.mean([r, g, b])

img = cv2.imread('Lenna.png')

set_rgb_avarage(img)