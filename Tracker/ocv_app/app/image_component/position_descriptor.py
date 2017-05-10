import cv2
import numpy as np

from . import img_processing as ipro
from ..math_component import operations as op
from .position_model import PositionModel

class PositionDescriptor(object):
    """docstring for PositionDescriptor"""
    
    def __init__(self):
        super(PositionDescriptor, self).__init__()

        