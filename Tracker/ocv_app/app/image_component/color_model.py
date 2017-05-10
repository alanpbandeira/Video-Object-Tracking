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
    def __init__(self, obj_features, bkgd_features, n_colors, qtz_obj):
        super(ColorModel, self).__init__()
        self.obj = qtz_obj
        self.obj_features = obj_features
        self.bkgd_features = bkgd_features
        self.obj_dim = (self.obj.shape[0], self.obj.shape[1])
        self.n_colors = n_colors
        self.bitmask_map = None
        self.rgb_avarage = None

        self.bkgd_hist = np.histogram(bkgd_features, n_colors)[0]
        self.obj_hist = np.histogram(obj_features, n_colors)[0]

        self.llr = op.log_likelihood_ratio(self.obj_hist, self.bkgd_hist, 0.01)
        self.set_bitmask_map()
        self.set_rgb_avarage()
        self.centroid = op.bitmask_centroid(self.bitmask_map)

    def set_bitmask_map(self):
        """
        Set a color value for each pixel depending on it's Log Likelihood Ratio (LLR) 
        and set a bitmask_map image of these colors in the object model.
        :return:
        """

        t = 0.8
        mask_data = []

        for data in self.obj_features:
            # if data >= len(self.llr):
            #     print(data)
            if self.llr[data] > t:
                mask_data.append([1.0, 1.0, 1.0])
            else:
                mask_data.append([0.0, 0.0, 0.0])

        mask_map = np.array(mask_data).reshape(
            (self.obj_dim[0], self.obj_dim[1], 3))

        self.bitmask_map = mask_map

    def set_rgb_avarage(self):
        """docstring"""

        b = np.mean(self.obj[:,:,0])
        g = np.mean(self.obj[:,:,1])
        r = np.mean(self.obj[:,:,2])

        self.rgb_avarage = np.mean([r, g, b])