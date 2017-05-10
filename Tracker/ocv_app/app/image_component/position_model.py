import numpy as np

from ..math_component import operations as op

class PositionModel(Object):
    """docstring"""

    def __init__(
        self, mean_radius, coords, obj_features, bkgd_feature, sectors):
        super(PositionModel, self).__init__()

        self.obj_coords = coords
        self.obj_features = obj_features
        self.bkgd_features = bkgd_feature
        self.bitmask = None

        self.obj_dim = (
            abs(obj_coords[0][0], obj_coords[1][0])
            abs(obj_coords[0][1], obj_coords[1][1])
        )

        self.mean_radius = mean_radius
        self.sectors = sectors

        self.obj_hist = np.histogram(self.obj_features, len(self.sectors))[0]
        self.bkgd_hist = np.histogram(self.bkgd_feature, len(self.sectors))[0]

        self.llr = op.log_likelihood_ratio(self.obj_hist, self.bkgd_hist, 0.01)

        self.set_bitmask_map()

    
    def set_bitmask_map(self):
        
        t = 0.08
        mask_data = []

        for pixel in self.obj_features:
            # if data >= len(self.llr):
            #     print(data)
            if self.llr[pixel] > t:
                mask_data.append([1.0, 1.0, 1.0])
            else:
                mask_data.append([0.0, 0.0, 0.0])

        mask_map = np.array(mask_data).reshape(
            (self.obj_dim[0], self.obj_dim[1], 3))

        self.bitmask_map = mask_map