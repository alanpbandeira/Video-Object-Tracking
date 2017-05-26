import math
import numpy as np


def calc_diag_rect(p_one, p_two):
    """DocString"""
    # width = abs(p_one[0] - p_two[0])
    # height = abs(p_one[1] - p_two[1])

    if p_one[0] < p_two[0] and p_one[1] < p_two[1]:
        top_idx = p_one
        bot_idx = p_two
    elif p_one[0] > p_two[0] and p_one[1] < p_two[1]:
        top_idx = (p_two[0], p_one[1])
        bot_idx = (p_one[0], p_two[1])
    elif p_one[0] < p_two[0] and p_one[1] > p_two[1]:
        top_idx = (p_one[0], p_two[1])
        bot_idx = (p_two[0], p_one[1])
    else:
        top_idx = p_two
        bot_idx = p_one

    # return origin[0], origin[1], width, height
    return top_idx, bot_idx

def calc_bkgd_rect(top_idx, bot_idx):
    """DocString"""
    width = abs(top_idx[0] - bot_idx[0])
    height = abs(top_idx[1] - bot_idx[1])

    delta = calc_delta(width, height)

    new_top = (top_idx[0] - delta, top_idx[1] - delta)
    new_bot = (bot_idx[0] + delta, bot_idx[1] + delta)

    bkgd_rect = calc_diag_rect(new_top, new_bot)

    return bkgd_rect, delta

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

def bitmask_centroid(bitmask_map):
    """docstring"""

    obj_idx = np.transpose(np.where(bitmask_map == 1))

    return tuple(np.int_(sum(obj_idx) // len(obj_idx)))

def pnt_dist(p_one, p_two):
    return np.ceil(
        np.sqrt(sum([np.power((a-b), 2) for a, b in zip(p_one, p_two)])))