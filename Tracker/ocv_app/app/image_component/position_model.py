import numpy as np
import cv2

class PositionModel(Object):
    """docstring"""

    def __init__(self, cent_radius, coords):
        super(PositionModel, self).__init__()

        # img coords
        # obj_feature_vector - sectors
        # bkgd_feature_vector - sectors
        # obj_dim = (x' - x, y'- y)
        # centroid_radius
        # number of sectors
        # (radius - sector) list

        # obj_hist
        # bkgd_hist

        # llr
        # bitmask
    
    def set_bitmask_map(self):
        pass