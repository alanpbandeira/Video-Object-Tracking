from collections import deque
import numpy as np
import argparse
import imutils
import cv2


# Arguments for function call
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the optional video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="Max buffer size")
args=vars(ap.parse_args())


if not args.get("video", False):
	camera = cv2.VideoCapture(0)
else:
	camera = cv2.VideoCapture(args["video"])

while True:
	# read a frame
	(grabbed, frame) = camera.read()

	if args.get("video") and not grabbed:
		break

	# Resize frame
	# Blur frame
	# Convert colors to hsv
	frame = imutils.resize(frame, width=600)
	
	cv2.imshow("frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()
