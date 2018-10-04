"""
capture pictures: 1 depth, 1 colour
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time

# standard
# Configure depth and color streams...
pipeline = rs.pipeline()
config = rs.config()
# config.enable_device('819112072219')
align_to = rs.stream.color
align = rs.align(align_to)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming from both cameras
profile = pipeline.start(config)

def draw_rectangle(img, top_left, bottom_right):
    """
    Draw a rectangle box around the given coordinates.
    top_left and bottom_right are tuples of ints that represent coordinates of vertices
    x1,y1 ------
    |          |
    |          |
    |          |
    --------x2,y
    """
    cv2.rectangle(img, top_left, bottom_right, (0,255,0), 2)

def get_distance():
    """
    return the average distance of a 10x10 box in the middle of the frame
    """
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    # coordinates specific to 640x480 frame
    # Convert images to numpy arrays
    # depth = np.asanyarray(depth_frame.get_data())
    # Crop depth data
    # depth = depth[310:330,230:250].astype(float)
    distance = 0
    for i in range(300, 340):
        for j in range(220, 260):
            distance += depth_frame.get_distance(i,j)
    return distance/1600

# streaming loop
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # set depth
        # color_image[depth_image>650]=255
        # depth_image[depth_image>650]=255
        # depth_image = depth_image*170

        # flip images
        # color_image = cv2.flip(color_image, 1 )
        # depth_image = cv2.flip(depth_image, 1 )

        # render depth colour
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack all images horizontally
        # images = np.hstack((color_image, depth_image))

        # Show images from both cameras
        window = cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        draw_rectangle(window, (300, 220), (340, 260))
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

        # Save images and depth maps from both cameras by pressing 's'
        # waitkey value is ascii
        ch = cv2.waitKey(25)
        if ch==115:
            #name_number = str(random.randint(0,9999))
            name_number = time.strftime("%Y%m%d-%H%M%S")
            print(name_number)
            img_name = name_number + "_image.jpg"
            depth_name = name_number + "_depth.jpg"
            distance = str(get_distance())
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL

            # draw boxes
            draw_rectangle(color_image, (300, 220), (340, 260))
            draw_rectangle(depth_image, (300, 220), (340, 260))

            # write distance to image
            cv2.putText(
                img=color_image,
               text=distance,
                org=(450, 450),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255))

            cv2.imwrite(img_name,color_image)
            cv2.imwrite(depth_name, depth_image)
            print("Save")
            # write distance to image


        if ch==113:
            break

finally:
    # Stop streaming
    pipeline.stop()
