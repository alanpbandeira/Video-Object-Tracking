import cv2
import imutils
import numpy as np

from .image_component import img_processing as ipro
from .image_component.color_descriptor import ColorDescriptor
from .actions import ActionHandler
from .math_component import operations as op


class Tracker(object):
    """docstring"""

    def __init__(self, args):
        super(Tracker, self).__init__()
        self.args = args
        self.frame = None

        self.dscpt = ColorDescriptor(bins=16)
        # self.dscpt = ColorDescriptor(clusterer="kmeans")
        # self.dscpt = ColorDescriptor(clusterer="mbkmeans")

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
                #     self.dscpt.patch_extract()
                #     self.dscpt.data_extract()
                #     print("done")
                # except:
                #     print("No object found!")

                self.dscpt.data_extract(self.frame)
                cv2.imshow('bitmask', self.dscpt.color_model.bitmask_map)
                print("done")

            if key == ord("t"):
                print("tracking...")
                camera.release()
                self.object_track()
                break

    def object_track(self):
        """docstring"""

        if not self.dscpt.color_model:
            print("No object model to track.")
            return

        # Set tracking camera
        t_camera = cv2.VideoCapture(self.args["video"])

        t_dscpt = ColorDescriptor(bins=16)
        # t_dscpt = ColorDescriptor(clusterer="kmeans")
        # t_dscpt = ColorDescriptor(clusterer="mbkmeans")

        frame_count = 1

        while True:
            (grabbed, t_frame) = t_camera.read()

            if self.args.get("video") and not grabbed:
                print("end of video")
                break

            t_frame = imutils.resize(t_frame, width=600)
            view_frame = np.copy(t_frame)

            # Run for the first frame
            if frame_count == 1:
                t_dscpt.slct_points = self.dscpt.slct_points.copy()
                t_dscpt.selections = self.dscpt.selections.copy()
                t_dscpt.bkgd_selections = self.dscpt.bkgd_selections.copy()
                t_dscpt.delta = self.dscpt.delta

                p = t_dscpt.slct_points[-2]
                q = t_dscpt.slct_points[-1]

                slct_pnts = op.calc_diag_rect(p, q)

                t_dscpt.data_extract(t_frame)

                cv2.rectangle(
                    view_frame, slct_pnts[0], slct_pnts[1], ((255, 0, 0)), 1)
            else:
                converged = False

                # while not converged:
                for x in range(5):
                    if converged:
                        break

                    # t_dscpt.data_extract(t_frame)

                    try:
                        t_dscpt.data_extract(t_frame)
                    except:
                        break

                    ctd_d = (

                        abs(t_dscpt.color_model.centroid[1] - 
                            self.dscpt.color_model.centroid[1]),

                        abs(t_dscpt.color_model.centroid[0] - 
                            self.dscpt.color_model.centroid[0])
                    )

                    if (ctd_d[0] <= 5 and ctd_d[1] <=5):
                        frame_count += 1
                        converged = True
                    else:
                        # print("not converged on frame " + str(frame_count))
                        slct_idx = self.center_selection(
                            t_dscpt.color_model.centroid, 
                            t_dscpt.selections)

                        t_dscpt.selections = [slct_idx]
                        t_dscpt.slct_points = [slct_idx[0], slct_idx[1]]

                        bkgd_selections = (
                            (
                                slct_idx[0][0] - self.dscpt.delta,
                                slct_idx[0][1] - self.dscpt.delta,
                            ),

                            (
                                slct_idx[1][0] + self.dscpt.delta,
                                slct_idx[1][1] + self.dscpt.delta,
                            )
                        )

                        t_dscpt.bkgd_selections = [bkgd_selections]

                p = t_dscpt.slct_points[-2]
                q = t_dscpt.slct_points[-1]

                slct_idx = self.center_selection(
                    t_dscpt.color_model.centroid, 
                    t_dscpt.selections)

                cv2.rectangle(
                    view_frame, slct_idx[0], slct_idx[1], 
                    ((255, 0, 0)), 1)
                
                # if self.model_update(t_dscpt, (0.05*256)):
                # if self.model_update(t_dscpt, 50):
                #     print("updated", frame_count)

            cv2.imshow('tracker', view_frame)

            if frame_count != 1:
                cv2.imshow('bitmask', t_dscpt.color_model.bitmask_map)

            frame_count +=1

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    def model_update(self, current_patcher, max_deviation):
        """docstring"""

        current_deviation = abs(
            self.dscpt.color_model.rgb_avarage - 
            current_patcher.color_model.rgb_avarage
        )

        if current_deviation > max_deviation:
            self.dscpt.color_model = current_patcher.color_model
            return True
        else:
            return False

    def center_selection(self, new_cent, selections):
        """docstring"""
        d_top = self.dscpt.color_model.centroid[0] - selections[0][0][1]
        d_bot = selections[0][1][1] - self.dscpt.color_model.centroid[0]

        d_left = self.dscpt.color_model.centroid[1] - selections[0][0][0]
        d_right = selections[0][1][0] - self.dscpt.color_model.centroid[1]

        top_idx = new_cent[1] - d_left, new_cent[0] - d_top
        bot_idx = new_cent[1] + d_right, new_cent[0] + d_bot

        return top_idx, bot_idx
