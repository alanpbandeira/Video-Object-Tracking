import numpy as np
import cv2

class PositionModel(Object):
    """docstring"""

    def __init__(self, cent_radius, coords):
        super(PositionModel, self).__init__()

        # centroid_radius
        # number of sectors
        # (radius - sector) list
        # feature_vector - sectors
        # img coords

        # obj_hist
        # bkgd_hist