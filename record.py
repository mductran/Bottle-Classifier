import numpy as np
import cv2
import pyrealsense2 as rs

class myCamera:

    def initial(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        self.depth_frame = frames.get_depth_frame()
        self.color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        self.depth_img = np.asanyarray(self.depth_frame.get_data())
        self.color_img = np.asanyarray(self.color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_img, alpha=0.03), cv2.COLORMAP_JET)

    def final(self):
        cv2.destroyAllWindows()
        self.pipeline.stop()

    def show_result(self):
        # Stack both images horizontally
        images = np.hstack((self.color_img, self.depth_colormap))

        cv2.imshow('IMAGE', images)
        if cv2.waitKey(1) == 27:
            return True
        else:
            return False

    def aspect_ratio(self, X):
        dict = {
            424:240,
            640:480,
            1280:720
        }
        return dict[X]


class Camera(myCamera):

    def __init__(self, X=640, FRAMERATE=30):

        Y = self.aspect_ratio(X)

        # Configure depth and color stream
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.depth, X, Y, rs.format.z16, FRAMERATE)
        self.config.enable_stream(rs.stream.color, X, Y, rs.format.bgr8, FRAMERATE)

        # Start streaming
        self.pipeline.start(self.config)

class Record(myCamera):

    def __init__(self, FILE, X=640, FRAMERATE=30):

        Y = self.aspect_ratio(X)

        # Configure depth and color stream
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.depth, X, Y, rs.format.z16, FRAMERATE)
        self.config.enable_stream(rs.stream.color, X, Y, rs.format.bgr8, FRAMERATE)

        self.config.enable_record_to_file(FILE)

        # Start streaming
        self.pipeline.start(self.config)

class Read(myCamera):

    def __init__(self, FILE, X=640, FRAMERATE=30):

        Y = self.aspect_ratio(X)

        # Configure depth and color stream
        self.pipeline = rs.pipeline()
        config = rs.config()

        rs.config.enable_device_from_file(config, FILE)

        config.enable_stream(rs.stream.depth, X, Y, rs.format.z16, FRAMERATE)
        config.enable_stream(rs.stream.color, X, Y, rs.format.bgr8, FRAMERATE)

        # Start streaming
        self.pipeline.start(config)

if __name__ == "__main__":
#######################################################################################################################
################################### \\ THESE IS AN EXAMPLE FOR USED THIS FUNCTION //###################################
#######################################################################################################################

    ''' This for open or use camera '''
    cam = Camera(640, 30)
    while True:
        cam.initial()
        if cam.show_result():
            cam.final()
            break

    ''' Thia for record to .bag file '''
    # rec = Record('test.bag', 424, 15)
    # t = time.time()
    # while True:
    #     rec.initial()
    #     print(time.time())
    #     if (time.time() - t) >= 5.00:
    #         rec.final()
    #         break

    ''' This for read .bsag file '''
    # red = Read('test.bag', 424, 15)
    # while True:
    #     red.initial()
    #     if red.show_result():
    #         red.final()
#         break