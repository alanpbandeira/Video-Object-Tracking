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

            sector_data = []

            for dist in dist_data:
                for sec_range in sorted(sectors.keys()):
                    if dist <= sec_range:
                        sector_data.append(sectors[sec_range])


    def set_sectors(self, max_dist):
        """docstring"""

        delta_sector = np.floor(max_dist / self.sectors)
        sector_range = max_dist
        sectors = {}
        
        for x in range(0, self.bins, -1):
            sectors[sector_range] = x
            sector_range -= delta_sector
        
        return sectors
