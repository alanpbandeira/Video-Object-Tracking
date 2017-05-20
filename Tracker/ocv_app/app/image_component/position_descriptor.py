import cv2
import numpy as np

from . import img_processing as ipro
from ..math_component import operations as op
from .position_model import PositionModel

class PositionDescriptor(object):
    """docstring for PositionDescriptor"""
    
    def __init__(self, bins):
        super(PositionDescriptor, self).__init__()

        # Delta width from object rect to bkgd rect
        self.delta = None
        # Extracted patches
        self.obj_data = None
        # Data from selection and background
        self.scn_data = None
        # Position models
        self.pos_model = None
        # Patch selected regions (points)
        self.selections = []
        # Patch selected regions with background (points)
        self.bkgd_selections = []

        # Circular sectors
        self.bins = bins
    
    def data_extract(self, img, centroid):
        """docstring"""

        for idx in range(len(self.bkgd_selections)):
            # Points enclosing background and object patch.
            # Points enclosing object patch. 
            scn_pnts = self.bkgd_selections[idx]
            obj_pnts = self.selections[idx]

            # Calculate object dimensions
            obj_w = obj_pnts[1][0] - obj_pnts[0][0]
            obj_h = obj_pnts[1][1] - obj_pnts[0][1]

            # Extract the scene (object + background) from the image.
            scn_roi = img[
                scn_pnts[0][1]:scn_pnts[1][1] + 1,
                scn_pnts[0][0]:scn_pnts[1][0] + 1
            ]

            indices = np.indices(scn_roi.shape[0], scn_roi.shape[1])
            indices = np.transpose(indices)
            pos_data = sorted([tuple(y) for x in indices for y in x])

            vfunc = np.vectorize(op.pnt_dist)
            dist_data = vfunc(pos_data, centroid)

            max_dist = max(dist_data)

            sectors = self.set_sectors(max_dist)
           
            radius = sum(dist_data) / len(dist_data)

            qnt_data = np.array([])

            for dist in dist_data:
                for sec_range in sorted(sectors.keys()):
                    if dist <= sec_range:
                        np.append(sector_data, sectors[sec_range])

            temp_data = qnt_data.reshape(scn_roi.shape[0], scn_roi.shape[1])

            patch_data = temp_data[
                self.delta:(self.delta + obj_h),
                self.delta:(self.delta + obj_w)
            ]

            patch_data = patch_data.reshape(obj_w*obj_h)

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

            self.pos_model = PositionModel(
                radius, obj_pnts, patch_data, bkgd_data)

    def bitmask_update(self, bitmaks, centroid):

        indices = np.indices(scn_roi.shape[0], scn_roi.shape[1])
        indices = np.transpose(indices)
        pos_data = sorted([tuple(y) for x in indices for y in x])

        point_dist_data = {}
        obj_pnts = [
            point for point in pos_data if np.array_equal(
                bitmaks[point], np.ones(3))
            ]

        dist_data = map(op.pnt_dist, obj_pnts)
        radius = sum(dist_data) / len(dist_data)
        new_obj_pnts = []

        for x in range(obj_pnts):
            if dist_data[x] <= radius:
                new_obj_pnts.append(obj_pnts[x])
        
        for point in obj_pnts:
            if point not in new_obj_pnts:
                centroid[point] = np.zeros(3)

    def set_sectors(self, max_dist):
        """docstring"""

        delta_sector = np.floor(max_dist / self.sectors)
        sector_range = max_dist
        sectors = {}
        
        for x in range(0, self.bins, -1):
            sectors[sector_range] = x
            sector_range -= delta_sector

]        return sectors
