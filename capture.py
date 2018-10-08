"""
capture pictures: 1 depth, 1 colour
once camera is up running:
    1. press 'a' to automatically capture 30 photos, 1 every 1.5 seconds
    2. press 's' to capture 1 photo
    3, press 'v' to record video
    4. press 'q' to quit camera
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time


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
    |           |
    |           |
    |           |
    --------x2,y2
    """
    cv2.rectangle(img, top_left, bottom_right, (0,255,0), 2)

def get_distance():
    """
    return the average distance of a 10x10 box in the middle of the frame
    """
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    distance = 0
    for i in range(300, 340):
        for j in range(220, 260):
            distance += depth_frame.get_distance(i,j)
    return distance/1600

def capture():
    """
    save a rgb and a depth photo.
    """
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # name_number = str(random.randint(0,9999))
    name_number = time.strftime("%Y%m%d-%H%M%S")
    print(name_number)
    img_name = "color" + name_number + ".jpg"
    depth_name = "depth" + name_number + ".jpg"
    distance = str(get_distance())
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    # write distance to image
    cv2.putText(
        img=color_image,
        text=distance,
        org=(450, 450),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(255, 255, 255))

    # draw_rectangle(color_image, (300, 220), (340, 260))
    # draw_rectangle(depth_image, (300, 220), (340, 260))

    cv2.imwrite(img_name, color_image)
    cv2.imwrite(depth_name, depth_image)
    print("Save")


# streaming loop
if __name__ == "__main__":
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

            # render depth colour
            depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Show images from both cameras
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            draw_rectangle(color_image, (300, 220), (340, 260))
            draw_rectangle(depth_image, (300, 220), (340, 260))
            cv2.imshow('RealSense', color_image)

            ch = cv2.waitKey(25)
            if ch == 97:
                for i in range(12):
                    capture()
                    time.sleep(3.0)

            if ch == 115:  # 's' for saving photos
                capture()
                continue

            if ch == 113:  # 'q' for quit
                break

    finally:
        # Stop streaming
        pipeline.stop()
