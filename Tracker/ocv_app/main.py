import argparse

from app.color_model_tracker import Tracker


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="Path to the optional video file")
args = vars(ap.parse_args())

app = Tracker(args)
app.run()