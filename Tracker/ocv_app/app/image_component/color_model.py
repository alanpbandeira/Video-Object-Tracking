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

        self.bkgd_hist = ipro.color_hist(bkgd_features, n_colors)
        self.obj_hist = ipro.color_hist(obj_features, n_colors)

        self.llr = op.log_likelihood_ratio(self.obj_hist, self.bkgd_hist, 0.01)

        self.bitmask_map, self.bitmask = ipro.set_bitmask_map(
            self.obj_features, self.llr, self.obj_d)

        self.centroid = op.bitmask_centroid(self.bitmask)

    def set_bitmask_map(self):
        """
        Set a color value for each pixel depending on it's 
        Log Likelihood Ratio (LLR) and set a bitmask_map 
        image of these colors in the object model.
        :return:
        """

        t = 0.5
        patch = self.obj_features.reshape((self.obj_d[0], self.obj_d[1], 3))
        t_idc = np.transpose(np.indices((self.obj_d[0], self.obj_d[1])))
        idc = [tuple(y) for x in t_idc for y in x]
        mask_data = np.zeros(
            (self.obj_d[0], self.obj_d[1], 3))
        i_bitmask = np.zeros((self.obj_d[0], self.obj_d[1]))

        colors = [tuple(np.int_(x)) for x in self.obj_features]

        for i in idc:
            # if data >= len(self.llr):
            #     print(data)
            if self.llr[tuple(np.int_(patch[i]))] > t:
                mask_data[i] = np.ones(3)
                i_bitmask[i] = 1

        self.bitmask_map = mask_data
        self.bitmask = i_bitmask