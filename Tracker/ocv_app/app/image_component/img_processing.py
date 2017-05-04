import cv2
import numpy as np

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
    """
    """
    indexes = {}
    flat_idx = []
    nxt_idx = 0

    q_range = 256 / bins
    q_img = image // q_range + 1
    
    for y in q_img:
        for x in y:
            if indexes.keys():
                if tuple(x) in indexes.keys():
                    continue
    
            indexes[tuple(x)] = nxt_idx
            nxt_idx += 1
    
    for y in q_img:
        for x in y:
            flat_idx.append(indexes[tuple(x)])
    
    return q_img, np.array(flat_idx), max(flat_idx)+1



