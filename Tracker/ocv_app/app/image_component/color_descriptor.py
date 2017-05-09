import cv2
import numpy as np

from . import img_processing as ipro
from ..math_component import operations as op
from .color_model import OBJPatch, ScenePatch, ColorModel


class ColorDescriptor(object):
    """docstring for CVIMGData."""

    def __init__(self, clusterer="simple", bins=8, colors=None):
        super(ColorDescriptor, self).__init__()
        # Video Frame
        self.img = None

        # Delta width from object rect to bkgd rect
        self.delta = None
        # Extracted patches
        self.obj_data = None
        # Data from selection and background
        self.scn_data = None
        # Color models
        self.color_model = None

        # Selected points
        self.slct_points = []
        # Patch selected regions (points)
        self.selections = []
        # Patch selected regions with background (points)
        self.bkgd_selections = []

        # Quantization technique
        self.clusterer = clusterer

        # Simple quantization bins
        self.bins = bins

        # Kmeans quantization colors
        self.colors = colors

    def load_img(self, img):
        """
        Store an image file from the provided parameter.
        :param img_path: path to the image file.
        :return:
        """
        self.img = np.copy(img)

    def patch_extract(self):
        """docstring"""

        if self.selections:

            for point in self.selections:
                obj_patch = self.img[
                    point[0][1]:point[1][1] + 1, 
                    point[0][0]:point[1][0] + 1]

                self.obj_data = OBJPatch(obj_patch, point)

    def data_extract(self):
        """docstring"""

        if self.selections:

            for idx in range(len(self.bkgd_selections)):
                # Points enclosing background and object patch.
                # Points enclosing object patch. 
                scn_pnts = self.bkgd_selections[idx]
                obj_pnts = self.selections[idx]

                # Calculate object dimensions
                obj_width = obj_pnts[1][0] - obj_pnts[0][0]
                obj_height = obj_pnts[1][1] - obj_pnts[0][1]

                # Calculate the delta width between object and background limit.
                delta = op.calc_delta(obj_width, obj_height)

                # Extract the scene (object + background) from the image.
                scn_roi = self.img[
                    scn_pnts[0][1]:scn_pnts[1][1] + 1,
                    scn_pnts[0][0]:scn_pnts[1][0] + 1
                ]

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
                temp_data = qnt_data.reshape(
                    scn_roi.shape[0], scn_roi.shape[1])

                # Extract the quantized object patch
                if qnt_roi is not None:
                    qtnz_obj_patch =  qnt_roi[
                        self.delta:(self.delta + obj_height),
                        self.delta:(self.delta + obj_width)
                    ]
                else:
                    qtnz_obj_patch = None

                # Extract objects quantized data
                patch_data = temp_data[
                    self.delta:(self.delta + obj_height),
                    self.delta:(self.delta + obj_width)
                ]

                patch_data = patch_data.reshape(obj_width*obj_height)

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

                # Store patch data and quantized versions
                # self.scn_data = ScenePatch(
                #     scn_pnts, self.delta, scn_roi, qnt_roi, qtnz_obj_patch)
                
                self.color_model = ColorModel(
                    patch_data, bkgd_data, self.colors, qtnz_obj_patch)