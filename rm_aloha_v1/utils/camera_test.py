import time

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device('REPLACE_WITH_LEFT_WRIST_CAMERA_SERIAL')  # 替换为相机1的序列号
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device('REPLACE_WITH_RIGHT_WRIST_CAMERA_SERIAL')  #
config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline_3 = rs.pipeline()
config_3 = rs.config()
config_3.enable_device('REPLACE_WITH_HIGH_CAMERA_SERIAL')  # 替换为相机3的序列号
config_3.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline_1.start(config_1)
pipeline_2.start(config_2)
pipeline_3.start(config_3)

for _ in range(20):
    frames_1 = pipeline_1.wait_for_frames()
    frames_2 = pipeline_2.wait_for_frames()
    frames_3 = pipeline_3.wait_for_frames()


def get_camera():
    # time.sleep(0.8)

    # Wait for a coherent pair of frames: depth and color
    frames_1 = pipeline_1.wait_for_frames()
    color_frame_1 = frames_1.get_color_frame()

    frames_2 = pipeline_2.wait_for_frames()
    color_frame_2 = frames_2.get_color_frame()

    frames_3 = pipeline_3.wait_for_frames()
    color_frame_3 = frames_3.get_color_frame()

    # if not color_frame_1 or not color_frame_2 or not color_frame_3:
    #     continue

    # Convert images to numpy arrays
    color_image_1 = np.asanyarray(color_frame_1.get_data())
    color_image_2 = np.asanyarray(color_frame_2.get_data())
    color_image_3 = np.asanyarray(color_frame_3.get_data())
    return color_image_1, color_image_2, color_image_3


if __name__ == '__main__':
    # for i in range(2):
    #     color_image_1, color_image_2, color_image_3 = get_camera()

    while True:
        # while True:
        t0 = time.time()
        # print('start',time.time())
        color_image_1, color_image_2, color_image_3 = get_camera()
        t1 = time.time()
        print('time:', t1 - t0)
        # time.sleep(1)
        # Stack all three frames horizontally for display
        images = np.hstack((color_image_1, color_image_2, color_image_3))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)
        # print([color_image_1, color_image_2, color_image_3])
