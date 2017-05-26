import cv2
import numpy as np

from collections import Counter
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import MiniBatchKMeans


def kmeans_qntz(image, centroids):
    """docstring"""
    
    image = np.float32(image) / 256

    # Shape the image into a matrix of pixels
    pixels = np.reshape(image, (image.shape[0] * image.shape[1], 3))

    # CLusterization using 8 colors (centroids)
    centroids, _ = kmeans(pixels, centroids)

    # Performs quantization
    qnt, _ = vq(pixels, centroids)

    # Reshape quantization
    centers_idx = np.reshape(qnt, (image.shape[0], image.shape[1]))
    clustered = centroids[centers_idx]

    clusteres = np.floor(clustered*256).astype('uint8')

    return clustered, qnt

def minibatch_kmeans(image, centroids):
    """docstring"""

    h, w = image.shape[:2]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    image = image.reshape((h * w, 3))

    clt = MiniBatchKMeans(centroids)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype('uint8')[labels]

    quant = quant.reshape((h, w, 3))

    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

    return quant, labels

def simple_qntz(image, bins):
    """docstring"""
    q_range = 256 / bins
    q_img = (image // q_range)

    return q_img

def color_hist(pixels, bins):
    """docstring"""
    hist = np.zeros((bins, bins, bins))

    pixels = [tuple(np.int_(x)) for x in pixels]

    for pixel in pixels:
        hist[pixel] = hist[pixel] + 1

    return hist

def set_bitmask_map(obj_data, llr, obj_d):
    """docstring"""

    t = 0.8
    patch = obj_data.reshape((obj_d[0], obj_d[1], 3))
    t_idc = np.transpose(np.indices((obj_d[0], obj_d[1])))
    idc = sorted([tuple(y) for x in t_idc for y in x])
    mask_data = np.zeros((obj_d[0], obj_d[1], 3))
    i_bitmask = np.zeros((obj_d[0], obj_d[1]))

    colors = [tuple(np.int_(x)) for x in obj_data]

    for i in idc:
        if llr[tuple(np.int_(patch[i]))] > t:
            mask_data[i] = np.ones(3)
            i_bitmask[i] = 1

    return mask_data, i_bitmask




