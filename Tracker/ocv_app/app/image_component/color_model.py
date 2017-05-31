import numpy as np

from ..math_component import operations as op
from . import img_processing as ipro

class OBJPatch(object):
    """docstring for IMGPatch."""
    def __init__(self, patch, coord):
        super(OBJPatch, self).__init__()
        self.coord = coord
        self.patch = patch


class ScenePatch(object):
    """docstring for BKGDPatch."""
    def __init__(self, coord, delta, scn_patch, qtz_scn, qtz_obj=None):
        super(ScenePatch, self).__init__()
        self.coord = coord
        self.delta = delta
        self.scn_patch = scn_patch
        self.qtz_scn = qtz_scn
        self.qtz_obj = qtz_obj


class ColorModel(object):
    """docstring for ColorModel"""
    def __init__(self, obj_features, bkgd_features, n_colors, obj_d):
        super(ColorModel, self).__init__()
        self.obj_features = obj_features
        self.bkgd_features = bkgd_features
        self.obj_d = obj_d
        self.n_colors = n_colors
        self.rgb_avarage = None

        self.bkgd_hist = ipro.nd_hist(3, n_colors, bkgd_features)
        self.obj_hist = ipro.nd_hist(3, n_colors, obj_features)

        self.llr = op.log_likelihood_ratio(self.obj_hist, self.bkgd_hist, 0.01)

        self.bitmask_map, self.bitmask = ipro.set_bitmask_map(
            self.obj_features, self.llr, self.obj_d)

        self.centroid = op.bitmask_centroid(self.bitmask)
