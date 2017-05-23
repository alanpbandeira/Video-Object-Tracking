import numpy as np

from ..math_component import operations as op

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
        self.bitmask_map = None
        self.rgb_avarage = None

        # self.bkgd_hist = np.histogram(bkgd_features, n_colors)[0]
        # self.obj_hist = np.histogram(obj_features, n_colors)[0]
        self.bkgd_hist = np.histogram(bkgd_features, max(bkgd_features))[0]
        self.obj_hist = np.histogram(obj_features, max(obj_features))[0]

        self.llr = op.log_likelihood_ratio(self.obj_hist, self.bkgd_hist, 0.01)
        self.set_bitmask_map()
        self.centroid = op.bitmask_centroid(self.bitmask_map)

    def set_bitmask_map(self):
        """
        Set a color value for each pixel depending on it's 
        Log Likelihood Ratio (LLR) and set a bitmask_map 
        image of these colors in the object model.
        :return:
        """

        t = 0.8
        mask_data = []

        for data in self.obj_features:
            # if data >= len(self.llr):
            #     print(data)
            if self.llr[data-1] > t:
                mask_data.append(np.ones(3))
            else:
                mask_data.append(np.zeros(3))

        mask_map = np.array(mask_data).reshape(
            (self.obj_d[0], self.obj_d[1], 3))

        self.bitmask_map = mask_map