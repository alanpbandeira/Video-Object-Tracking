import cv2
import numpy as np

from . import img_processing as ipro
from ..math_component import operations as op
from .position_model import PositionModel
from .color_model import ColorModel
from .cp_model import CPModel

class CPDescriptor(object):
    """docstring for PositionDescriptor"""

    def __init__(self, clusterer="simple", bins=8, colors=None):
        super(PositionDescriptor, self).__init__()

        # Delta width from object rect to bkgd rect
        self.delta = None
        # Color Position models
        self.cp_model = None
        # Patch selected regions (points)
        self.selections = []
        # Patch selected regions with background (points)
        self.bkgd_selections = []

        # Color Quantization technique
        self.clusterer = clusterer

        # Simple quantization bins
        self.bins = bins

        # Kmeans quantization colors
        self.colors = colors

    def data_extract(self, img):
        for idx in range(len(self.bkgd_selections)):
            # Points enclosing background and object patch.
            # Points enclosing object patch.
            scn_pnts = self.bkgd_selections[idx]
            obj_pnts = self.selections[idx]

            # Calculate object dimensions
            obj_d = (
                obj_pnts[1][0] - obj_pnts[0][0],
                obj_pnts[1][1] - obj_pnts[0][1]
            )

            scn_d = (
                scn_pnts[1][0] - scn_pnts[0][0],
                scn_pnts[1][1] - scn_pnts[0][1]
            )

            # Extract the scene (object + background) from the image.
            scn_roi = img[
                scn_pnts[0][1]:scn_pnts[1][1] + 1,
                scn_pnts[0][0]:scn_pnts[1][0] + 1
            ]

            self.build_color_model(scn_roi, obj_d)
            self.build_pos_model(scn_d, obj_d, self.color_model.centroid)

    def build_pos_model(self, scn_d, obj_d, centroid):
        """docstring"""

        indices = np.indices(scn_d[0], scn_d[1])
        indices = np.transpose(indices)
        pos_data = sorted([tuple(y) for x in indices for y in x])

        dist_data = np.array([op.pnt_dist(pos, centroid) for pos in pos_data])

        temp_data = dist_data.reshape((scn_d[0], scn_d[1]))

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

        self.pos_model = PositionModel(obj_d, patch_data, bkgd_data)

    def build_color_model(self, scn_roi, obj_d):
        """docstring"""

        if self.clusterer == "kmeans":
            qnt_roi, qnt_data = ipro.kmeans_qntz(scn_roi, self.colors)
        elif self.clusterer == "mbkmeans":
            qnt_roi, qnt_data = ipro.minibatch_kmeans(
                scn_roi, self.colors)
        else:
            # Generate the quantized scene patch,
            # its' quantization data and number of colors.
            qnt_roi, qnt_data, self.colors = ipro.simple_qntz(
                scn_roi, self.bins)

        # Shape the quantized scene data to scenes shape
        temp_data = qnt_data.reshape(scn_roi.shape[0], scn_roi.shape[1])

        # Extract the quantized object patch
        if qnt_roi is not None:
            qtnz_obj_patch =  qnt_roi[
                self.delta:(self.delta + obj_d[0]),
                self.delta:(self.delta + obj_d[1])
            ]
        else:
            qtnz_obj_patch = None

        # Extract objects quantized data
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

        self.color_model = ColorModel(
            patch_data, bkgd_data, self.colors, qtnz_obj_patch)
