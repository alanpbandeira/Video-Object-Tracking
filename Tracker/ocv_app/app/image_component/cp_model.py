import numpy as np

from ..math_component import operations as op
from . import img_processing as ipro

class CPModel(object):
    def __init__(self, obj_features, bkgd_features, bin_size, obj_d):
        super(CPModel, self).__init__()
        
        self.objd_features = obj_features
        self.bkgd_features = bkgd_features
        self.obj_d = obj_d
        # self.rgb_avarage = None

        self.bkgd_hist = ipro.nd_hist(4, bin_size, bkgd_features)
        self.obj_hist = ipro.nd_hist(4, bin_size, obj_features)

        self.llr = op.log_likelihood_ratio(self.obj_hist, self.bkgd_hist, 0.01)

        self.bitmask_map, self.bitmask = ipro.set_bitmask_map(
            self.obj_features, self.llr, self.obj_d)

        self.centroid = op.bitmask_centroid(self.bitmask)
