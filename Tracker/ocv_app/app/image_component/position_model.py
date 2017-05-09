import numpy as np
import cv2

class PositionModel(Object):
    """docstring"""

    def __init__(
        self, mean_radius, coords, obj_features, bkgd_feature, sectors):
        super(PositionModel, self).__init__()

        self.obj_coords = coords
        self.obj_features = obj_features
        self.bkgd_features = bkgd_feature

        self.obj_dim = (
            abs(obj_coords[0][0], obj_coords[1][0])
            abs(obj_coords[0][1], obj_coords[1][1])
        )

        self.mean_radius = mean_radius
        self.sectors = sectors
        # (radius - sector) list

        # obj_hist
        # bkgd_hist

        # llr
        # bitmask
    
    def set_bitmask_map(self):
        pass