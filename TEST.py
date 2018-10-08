import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
pipeline.start()

# streaming loop
while True:
    # block program till frames arrive
    frames = pipeline.wait_for_frames()

    # get depth image
    depth_frame = frames.get_depth_frame()

    # query distance to object in center of image
    center_distance = depth_frame.get_distance(320, 240)
    print(center_distance)