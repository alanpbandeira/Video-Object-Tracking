import numpy as np


class CPModel(object):
    def __init__(self, llr):
        super(CPModel, self).__init__()
        self.llr = llr
        self.bitmask_map = None
