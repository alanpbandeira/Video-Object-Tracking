import img_processing as ipro
import cv2
import numpy as np
import math


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

def calc_delta(width, height):
    """DocString"""
    return math.ceil(((width + height) / 4) * (math.sqrt(2) - 1))

def log_likelihood_ratio(obj_hist, bkgd_hist, v_error):
    """DocString"""

    idx = np.transpose(np.where(obj_hist == 0))
    z_idx = [tuple(np.int_(x)) for x in idx]
    for idx in z_idx:
        obj_hist[idx] = v_error

    idx = np.transpose(np.where(bkgd_hist == 0))
    z_idx = [tuple(np.int_(x)) for x in idx]
    for idx in z_idx:
        bkgd_hist[idx] = v_error
    
    div = np.divide(obj_hist, bkgd_hist)
    log = np.log(div)

    return log

def set_bitmask_map(obj_data, llr, obj_d):
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

def bitmask_centroid(bitmask_map):
    """docstring"""

    obj_idx = np.transpose(np.where(bitmask_map == 1))

    return tuple(np.int_(sum(obj_idx) // len(obj_idx)))

obj_pnts = ((200, 200), (350, 400))
h = obj_pnts[1][1] - obj_pnts[0][1]
w = obj_pnts[1][0] - obj_pnts[0][0]
delta = calc_delta(w, h)
scn_pnts = (
    (
        obj_pnts[0][0] - delta,
        obj_pnts[0][1] - delta
    ),
    (
        obj_pnts[1][0] + delta,
        obj_pnts[1][1] + delta
    )
)

image = cv2.imread("Lenna.png")
qnt = simple_qntz(image, 8)

patch = qnt[
    obj_pnts[0][1]:obj_pnts[1][1], 
    obj_pnts[0][0]:obj_pnts[1][0] 
]

scn = qnt[
    scn_pnts[0][1]:scn_pnts[1][1], 
    scn_pnts[0][0]:scn_pnts[1][0] 
]

obj_data = patch.reshape((patch.shape[0] * patch.shape[1], 3))

top = scn[ :delta + 1, : ]
bot = scn[ scn.shape[0] - delta:, :]

left = scn[ 
    delta:scn.shape[0] - delta + 1, :delta ]

right = scn[ 
    delta:scn.shape[0] - delta + 1, scn.shape[1] - delta: ]

top = top.reshape((top.shape[0] * top.shape[1], 3))
bot = bot.reshape((bot.shape[0] * bot.shape[1], 3))
left = left.reshape((left.shape[0] * left.shape[1], 3))
right = right.reshape((right.shape[0] * right.shape[1], 3))

# bkgd_data = np.concatenate((top, bot, left, right))
bkgd_data = np.vstack((top, bot, left, right))

obj_hist = color_hist(obj_data, 8)
bkgd_hist = color_hist(bkgd_data, 8)

llr = log_likelihood_ratio(obj_hist, bkgd_hist, 0.01)

bitmask, i_bitmask = set_bitmask_map(obj_data, llr, (h,w))

print(bitmask_centroid(i_bitmask))

cv2.imshow('image', bitmask)
cv2.waitKey(0)
cv2.imshow('image', patch)
cv2.waitKey(0)