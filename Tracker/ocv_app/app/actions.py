import cv2

from .math_component import operations as op


class ActionHandler(object):
    
    def __init__(self, parent):
        super(ActionHandler, self).__init__()
        self.parent = parent

    def point_mark(self, event, x, y, flags, param):
        if event == cv2.EVENT_FLAG_LBUTTON:
            if (x, y) not in self.parent.patcher.slct_points:
                self.parent.patcher.slct_points.append((x, y))
                if len(self.parent.patcher.slct_points) % 2 == 0:
                    cv2.circle(self.parent.frame, (x, y), 3, (255, 0, 0), -1)
                    self.draw_selection()
                    cv2.imshow('window', self.parent.frame)
                else:
                    cv2.circle(self.parent.frame, (x, y), 3, (255, 0, 0), -1)
                    cv2.imshow('window', self.parent.frame)
            else:
                return
    
    def draw_selection(self):
         p = self.parent.patcher.slct_points[-2]
         q = self.parent.patcher.slct_points[-1]

         slct_pnts = op.calc_diag_rect(p, q)
         bkgd_pnts, self.parent.patcher.delta = op.calc_bkgd_rect(
             slct_pnts[0], slct_pnts[1])
         
         if slct_pnts in self.parent.patcher.selections:
             return
         
         cv2.rectangle(
             self.parent.frame, slct_pnts[0], slct_pnts[1], ((255, 0, 0)), 1)
         cv2.rectangle(
             self.parent.frame, bkgd_pnts[0], bkgd_pnts[1], ((0, 255, 0)), 1)
         
         self.parent.patcher.selections.append(slct_pnts)
         self.parent.patcher.bkgd_selections.append(bkgd_pnts)
         
