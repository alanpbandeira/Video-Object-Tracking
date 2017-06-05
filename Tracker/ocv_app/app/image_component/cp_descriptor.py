import cv2
import numpy as np

from . import img_processing as ipro
from ..math_component import operations as op
from .color_model import ColorModel
from .cp_model import CPModel

class CPDescriptor(object):
    """docstring for PositionDescriptor"""

    def __init__(self, clusterer="simple", bins=8, colors=None):
        super(PositionDescriptor, self).__init__()

        # Delta width from object rect to bkgd rect
        self.delta = None
        # Color model
        self.color_model = None
        # Color Position models
        self.cp_model = None
        # Patch selected regions (points)
        self.selections = []
        # Patch selected regions with background (points)
        self.bkgd_selections = []

        # Scene matrix indexes
        self.scn_idx = None

        # Color Quantization technique
        self.clusterer = clusterer

        # Simple quantization bins and indexes
        self.bins = bins

    def data_extract(self, img):
        """docstring"""
        for idx in range(len(self.bkgd_selections)):
            # Points enclosing background and object patch.
            # Points enclosing object patch.
            scn_pnts = self.bkgd_selections[idx]
            obj_pnts = self.selections[idx]

            # Calculate object dimensions
            obj_d = (
                obj_pnts[1][1] - obj_pnts[0][1],
                obj_pnts[1][0] - obj_pnts[0][0]
            )

            scn_d = (
                scn_pnts[1][1] - scn_pnts[0][1],
                scn_pnts[1][0] - scn_pnts[0][0]
            )

            if self.scn_idx is None:
                self.scn_idx = ipro.get_img_idx(scn_d)

            self.build_color_model(img, obj_pnts, scn_pnts)

            obj_distances, bkgd_distances = self.postion_features(
                scn_d, obj_d, self.color_model.centroid)

            # TODO build Color-Position Model

    def postion_features(self, scn_d, obj_d, centroid):
        """docstring"""

        dist_data = np.array(
            [op.pnt_dist(pos, centroid) for pos in self.scn_idx])

        qnt_data = self.postion_indexation(dist_data)
        temp_data = qnt_data.reshape((scn_d[0], scn_d[1]))

        patch_data = temp_data[
            self.delta:(self.delta + obj_d[0]),
            self.delta:(self.delta + obj_d[1])
        ]

        patch_data = patch_data.reshape(obj_d[1] * obj_d[0])

        # Extract backgrounds quantized data
        top = temp_data[ :self.delta + 1, : ]
        bot = temp_data[ temp_data.shape[0] - self.delta:, :]

        left = temp_data[
            self.delta:temp_data.shape[0] - self.delta + 1,
            :self.delta ]

        right = temp_data[
            self.delta:temp_data.shape[0] - self.delta + 1,
            temp_data.shape[1] - self.delta: ]

        top = top.reshape(top.shape[0]*top.shape[1])
        bot = bot.reshape(bot.shape[0]*bot.shape[1])
        left = left.reshape(left.shape[0]*left.shape[1])
        right = right.reshape(right.shape[0]*right.shape[1])

        bkgd_data = np.concatenate((top, bot, left, right))

        return patch_data, bkgd_data

    def build_color_model(self, img, obj_pnts, scn_pnts):
        """docstring"""

        if self.selections:
            # Points enclosing background and object patch.
            # Points enclosing object patch.

            # Calculate object dimensions
            obj_w = obj_pnts[1][0] - obj_pnts[0][0]
            obj_h = obj_pnts[1][1] - obj_pnts[0][1]

            if self.clusterer == "kmeans":
                qnt_roi, qnt_data = ipro.kmeans_qntz(img, self.bins)
            elif self.clusterer == "mbkmeans":
                qnt_roi, qnt_data = ipro.minibatch_kmeans(
                    img, self.bins)
            else:
                # Generate the quantized scene patch,
                # its' quantization data and number of colors.
                qnt_img = ipro.simple_qntz(256, self.bins, img)

            scn_data = qnt_img[
                scn_pnts[0][1]:scn_pnts[1][1] + 1,
                scn_pnts[0][0]:scn_pnts[1][0] + 1
            ]

            # Extract objects quantized data
            patch_data = scn_data[
                self.delta:(self.delta + obj_h),
                self.delta:(self.delta + obj_w)
            ]

            patch_data = patch_data.reshape((obj_w * obj_h, 3))

            # Extract backgrounds quantized data
            top = scn_data[ :self.delta + 1, : ]
            bot = scn_data[ scn_data.shape[0] - self.delta:, :]

            left = scn_data[
                self.delta:scn_data.shape[0] - self.delta + 1,
                :self.delta ]

            right = scn_data[
                self.delta:scn_data.shape[0] - self.delta + 1,
                scn_data.shape[1] - self.delta: ]

            top = top.reshape((top.shape[0] * top.shape[1], 3))
            bot = bot.reshape((bot.shape[0] * bot.shape[1], 3))
            left = left.reshape((left.shape[0] * left.shape[1], 3))
            right = right.reshape((right.shape[0] * right.shape[1], 3))

            bkgd_data = np.vstack((top, bot, left, right))

            self.color_model = ColorModel(
                patch_data, bkgd_data, self.bins, (obj_h, obj_w))

            self.obj_avcol(obj_pnts, img)

    def obj_avcol(self, obj_pnts, img):
        """
        Claculates the detected object avarage rgb color
        """

        tmp_bm = np.zeros((img.shape[0], img.shape[1]))
        tmp_bm[
            obj_pnts[0][1]:obj_pnts[1][1],
            obj_pnts[0][0]:obj_pnts[1][0]
        ] = self.color_model.bitmask[:,:]

        bm_indices = np.transpose(np.where(tmp_bm == 1))
        bm_indices = [tuple(np.int_(x)) for x in bm_indices]

        pixels = [img[idx] for idx in bm_indices]

        pixels = np.vstack(pixels)

        b = np.mean(pixels[:,0])
        g = np.mean(pixels[:,1])
        r = np.mean(pixels[:,2])

        self.color_model.rgb_avarage = np.mean([r, g, b])

    def postion_indexation(self, scn_dist):
        upper_bound = max(data_dist)
        delta_d = upper_bound / self.bins

        indexes = list(range(self.bins))
        ranges = [delta_d * (i+1)) for i in indexes]
        qnt_id = dict(zip(ranges, indexes))

        idx_data = [qnt_id[r] for d in scn_dist for r in ranges if d <= r]

        return np.array(idx_data)
