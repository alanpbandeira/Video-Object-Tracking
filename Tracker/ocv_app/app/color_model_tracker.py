import cv2
import imutils
import numpy as np

from .image_component import img_processing as ipro
from .image_component.img_data import Patcher
from .actions import ActionHandler
from .math_component import operations as op


class Tracker(object):

    def __init__(self, args):
        super(Tracker, self).__init__()
        self.args = args
        self.frame = None

        self.patcher = Patcher(bins=16)
        # self.patcher = Patcher(clusterer="kmeans", colors=16)
        # self.patcher = Patcher(clusterer="mbkmeans", colors=16)
        
        self.act_hand = ActionHandler(self)

        self.camera_switch = False
        self.video_switch = False

        cv2.namedWindow('window')
        cv2.namedWindow('tracker')
        cv2.namedWindow('bitmask')

    def run(self):
        if not self.args.get("video", False):
            camera = cv2.VideoCapture(0)
            self.camera_switch = True
        else:
            camera = cv2.VideoCapture(self.args["video"])
            self.video_switch = True

        (grabbed, self.frame) = camera.read()

        if self.args.get("video") and not grabbed:
            return
        
        self.frame = imutils.resize(self.frame, width=600)
        self.patcher.load_img(self.frame)
        
        cv2.imshow('window', self.frame)
                
        while True:
            
            key = cv2.waitKey(0)

            # If the 'q' key is pressed, stop the loop
            if key == ord("q"):
                camera.release()
                cv2.destroyAllWindows()
                break
            
            # If the 's' key is pressed will be started the selection mode.
            if key == ord("s"):
                cv2.setMouseCallback('window', self.act_hand.point_mark)
            
            if key == ord("p"):
                # try:
                #     self.patcher.patch_extract()
                #     self.patcher.data_extract()
                #     print("done")
                # except:
                #     print("No object found!")
                
                self.patcher.patch_extract()
                self.patcher.data_extract()
                print("done")
            
            if key == ord("t"):
                print("tracking...")
                camera.release()
                self.object_track()
                break
    
    def object_track(self):
        """docstring"""
        
        if not self.patcher.color_model:
            print("No object model to track.")
            return
                
        # Set tracking camera
        t_camera = cv2.VideoCapture(self.args["video"])
        
        t_patcher = Patcher(bins=16)
        # t_patcher = Patcher(clusterer="kmeans", colors=16)
        # t_patcher = Patcher(clusterer="mbkmeans", colors=16)
        
        frame_count = 1

        while True:
            # print("new frame")
            (grabbed, t_frame) = t_camera.read()
            
            if self.args.get("video") and not grabbed:
                print("end of video")
                break
            
            t_frame = imutils.resize(t_frame, width=600)
            t_patcher.load_img(t_frame)
            view_frame = np.copy(t_frame)

            # Run for the first frame
            if frame_count == 1:

                t_patcher.slct_points = self.patcher.slct_points.copy()
                t_patcher.selections = self.patcher.selections.copy()
                t_patcher.bkgd_selections = self.patcher.bkgd_selections.copy()
                t_patcher.delta = self.patcher.delta
                
                p = t_patcher.slct_points[-2]
                q = t_patcher.slct_points[-1]

                slct_pnts = op.calc_diag_rect(p, q)

                t_patcher.patch_extract()
                t_patcher.data_extract()

                cv2.rectangle(
                    view_frame, slct_pnts[0], slct_pnts[1], ((255, 0, 0)), 1)
                
            else: 
                converged = False

                # while not converged:
                for x in range(5):
                    if converged:
                        break

                    try:
                        t_patcher.patch_extract()
                        t_patcher.data_extract()
                    except:
                        break
                    
                    # ctd_d = op.pnt_dist(
                    #     t_patcher.color_model.centroid,
                    #     self.patcher.color_model.centroid
                    # )

                    ctd_d = (
                        abs(t_patcher.color_model.centroid[0] - self.patcher.color_model.centroid[0]),
                        abs(t_patcher.color_model.centroid[1] - self.patcher.color_model.centroid[1]),
                    )

                    if (ctd_d[0] <= 5 and ctd_d[1] <=5):
                        frame_count += 1
                        converged = True
                    else:
                        # print("not converged on frame " + str(frame_count))
                        slct_idx = self.center_selection(
                            t_patcher.color_model.centroid, 
                            t_patcher.selections)
                        
                        t_patcher.selections = [slct_idx]
                        t_patcher.slct_points = [slct_idx[0], slct_idx[1]]

                        bkgd_selections = (
                            (
                                slct_idx[0][0] - self.patcher.delta,
                                slct_idx[0][1] - self.patcher.delta,
                            ),

                            (
                                slct_idx[1][0] + self.patcher.delta,
                                slct_idx[1][1] + self.patcher.delta,
                            )
                        )

                        t_patcher.bkgd_selections = [bkgd_selections]

                p = t_patcher.slct_points[-2]
                q = t_patcher.slct_points[-1]

                slct_idx = self.center_selection(
                    t_patcher.color_model.centroid, 
                    t_patcher.selections)

                cv2.rectangle(
                    view_frame, slct_idx[0], slct_idx[1], 
                    ((255, 0, 0)), 1)
                
                if self.model_update(t_patcher, (0.05*256)):
                    print("updated", frame_count)

            cv2.imshow('tracker', view_frame)

            if frame_count != 1:
                cv2.imshow('bitmask', t_patcher.color_model.bitmask_map)

            frame_count +=1

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    def model_update(self, current_patcher, max_deviation):
        """docstring"""

        current_deviation = abs(
            self.patcher.color_model.rgb_avarage - 
            current_patcher.color_model.rgb_avarage
        )

        if current_deviation > max_deviation:
            self.patcher.color_model = current_patcher.color_model
            return True
        else:
            return False

    def center_selection(self, new_cent, selections):
        """docstring"""
        d_top = self.patcher.color_model.centroid[1] - selections[0][0][1]
        d_bot = selections[0][1][1] - self.patcher.color_model.centroid[1]

        d_left = self.patcher.color_model.centroid[0] - selections[0][0][0]
        d_right = selections[0][1][0] - self.patcher.color_model.centroid[0]

        top_idx = new_cent[0] - d_left, new_cent[1] - d_top
        bot_idx = new_cent[0] + d_right, new_cent[1] + d_bot

        return top_idx, bot_idx
