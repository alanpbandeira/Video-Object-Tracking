import numpy as np

from ..math_component import operations as op

class PositionModel(Object):
    """docstring"""

    def __init__(self, obj_features, bkgd_feature):
        super(PositionModel, self).__init__()

        self.obj_features = obj_features
        self.bkgd_features = bkgd_feature

        self.obj_hist = np.histogram(self.obj_features, len(self.sectors))[0]
        self.bkgd_hist = np.histogram(self.bkgd_feature, len(self.sectors))[0]
        self.llr = np.array(
            op.log_likelihood_ratio(self.obj_hist, self.bkgd_hist, 0.01))