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

    llr = []

    for x in range(len(obj_hist)):
        ratio = max([obj_hist[x], v_error]) / max([bkgd_hist[x], v_error])
        llr.append(math.log(ratio))

    return llr

def bitmask_centroid(bitmask_map):
    x_coords = []
    y_coords = []

    for y in range(bitmask_map.shape[0]):
        for x in range(bitmask_map.shape[1]):
            if np.array_equal(bitmask_map[y][x], [1.0, 1.0, 1.0]):
                x_coords.append(x)
                y_coords.append(y)
            else:
                continue
                
    cent_x = math.ceil(sum(x_coords)/len(x_coords))
    cent_y = math.ceil(sum(y_coords)/len(y_coords))

    return cent_x, cent_y

def pnt_dist(p_one, p_two):
    return np.ceil(
        np.sqrt(sum([np.power((a-b), 2) for a, b in zip(p_one, p_two)])))