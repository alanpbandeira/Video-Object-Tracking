import cv2
import numpy as np

from . import img_processing as ipro
from ..math_component import operations as op
from .color_model import ColorModel


class ColorDescriptor(object):
    """docstring for ColorDescriptor."""

    def __init__(self, clusterer="simple", bins=8, colors=None):
        super(ColorDescriptor, self).__init__()

        # Delta width from object rect to bkgd rect
        self.delta = None
        # Color Model
        self.color_model = None
        # Selected points
        self.slct_points = []
        # Patch selected regions (points)
        self.selections = []
        # Patch selected regions with background (points)
        self.bkgd_selections = []

        # Quantization technique
        self.clusterer = clusterer

        # Simple quantization bins and indexes
        self.bins = bins
        self.qnt_info = None

        # Kmeans quantization colors
        self.colors = colors

    def data_extract(self, img):
        """docstring"""

        if self.selections:
            for idx in range(len(self.bkgd_selections)):
                # Points enclosing background and object patch.
                # Points enclosing object patch. 
                scn_pnts = self.bkgd_selections[idx]
                obj_pnts = self.selections[idx]

                # Calculate object dimensions
                obj_w = obj_pnts[1][0] - obj_pnts[0][0]
                obj_h = obj_pnts[1][1] - obj_pnts[0][1]
                
                # Extract the scene (object + background) from the image.
                # scn_roi = qnt_img[
                #     scn_pnts[0][1]:scn_pnts[1][1] + 1,
                #     scn_pnts[0][0]:scn_pnts[1][0] + 1
                # ]

                if self.clusterer == "kmeans":
                    qnt_roi, qnt_data = ipro.kmeans_qntz(img, self.colors)
                elif self.clusterer == "mbkmeans":
                    qnt_roi, qnt_data = ipro.minibatch_kmeans(
                        img, self.colors)
                else:
                    # Generate the quantized scene patch,
                    # its' quantization data and number of colors.
                    print("here 1")
                    qnt_img = ipro.simple_qntz(img, self.bins)

                    # self.colors = max(qnt_data) + 1


                # Shape the quantized scene data to scenes shape                
                # img_data = qnt_data.reshape(img.shape[0], img.shape[1])

                # Extract the quantized object patch
                # if qnt_roi is not None:
                #     qtnz_obj_patch =  qnt_roi[
                #         self.delta:(self.delta + obj_h),
                #         self.delta:(self.delta + obj_w)
                #     ]
                # else:
                #     qtnz_obj_patch = None
                scn_data = qnt_img[
                    scn_pnts[1][0]:scn_pnts[1][1],
                    scn_pnts[0][0]:scn_pnts[1][0]
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

                # bkgd_data = np.concatenate((top, bot, left, right))
                bkgd_data = np.vstack((top, bot, left, right))

                self.color_model = ColorModel(
                    patch_data, bkgd_data, self.colors, (obj_h, obj_w))
                
                obj_patch = img[
                    obj_pnts[0][1]:obj_pnts[1][1],
                    obj_pnts[0][0]:obj_pnts[1][0]
                ]

                self.obj_avcol(obj_patch)

    def obj_avcol(self, obj_patch):
        """
        Claculates the detected object avarage rgb color
        """
        h = obj_patch.shape[0]
        w = obj_patch.shape[1]
        indices = np.transpose(np.indices((h,w)))
        pos_data = sorted([tuple(y) for x in indices for y in x])

        pixels = []

        for pos in pos_data:
            if np.array_equal(self.color_model.bitmask_map[pos], np.ones(3)):
                pixels.append(obj_patch[pos])

        pixels = np.array(pixels)
        
        b = np.mean(pixels[:,0])
        g = np.mean(pixels[:,1])
        r = np.mean(pixels[:,2])

        self.color_model.rgb_avarage = np.mean([r, g, b])