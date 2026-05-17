# import time
#
# import pyrealsense2 as rs
# import numpy as np
# import cv2


# def camera_init():
#     pipeline_1 = rs.pipeline()
#     config_1 = rs.config()
#     config_1.enable_device('REPLACE_WITH_HIGH_CAMERA_SERIAL')  # high
#     config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#
#     pipeline_2 = rs.pipeline()
#     config_2 = rs.config()
#     config_2.enable_device('REPLACE_WITH_LEFT_WRIST_CAMERA_SERIAL')  # left
#     config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#
#     pipeline_3 = rs.pipeline()
#     config_3 = rs.config()
#     config_3.enable_device('REPLACE_WITH_RIGHT_WRIST_CAMERA_SERIAL')  # right
#     config_3.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#
#     # Start streaming
#     pipeline_1.start(config_1)
#     pipeline_2.start(config_2)
#     pipeline_3.start(config_3)
#
#     for _ in range(50):
#         frames_1 = pipeline_1.wait_for_frames()
#         frames_2 = pipeline_2.wait_for_frames()
#         frames_3 = pipeline_3.wait_for_frames()
#
#     return pipeline_1, pipeline_2, pipeline_3
#
#
# def get_camera(camera_name):
#     # time.sleep(0.8)
#     image_dict = dict()
#     camera_names = ['camera_high', 'camera_left', 'camera_right']
#     pipeline_1, pipeline_2, pipeline_3 = camera_name
#     # Wait for a coherent pair of frames: depth and color
#     frames_1 = pipeline_1.wait_for_frames()
#     color_frame_1 = frames_1.get_color_frame()
#
#     frames_2 = pipeline_2.wait_for_frames()
#     color_frame_2 = frames_2.get_color_frame()
#
#     frames_3 = pipeline_3.wait_for_frames()
#     color_frame_3 = frames_3.get_color_frame()
#
#     # if not color_frame_1 or not color_frame_2 or not color_frame_3:
#     #     continue
#
#     # Convert images to numpy arrays
#     color_image = []
#     color_image.append(np.asanyarray(color_frame_1.get_data()))
#     color_image.append(np.asanyarray(color_frame_2.get_data()))
#     color_image.append(np.asanyarray(color_frame_3.get_data()))
#
#     for i, cam_name in enumerate(camera_names):
#         image_dict[cam_name] = color_image[i]
#     return image_dict
# return [color_image_1, color_image_2, color_image_3]


import pyrealsense2 as rs
import numpy as np
import threading
import time
import cv2
import os
import copy

import logging
logging.basicConfig(level=logging.DEBUG)
class MultiCameraReader:
    def __init__(self, serial_numbers):
        self.pipelines = []
        self.frames = {}
        self.lock = threading.RLock()
        self.running = True
        self.camera_names = [ 'cam_left_wrist', 'cam_right_wrist']
        self.timestamps = {}

        # Initialize cameras
        for serial_number in serial_numbers:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial_number)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            self.pipelines.append(pipeline)
            # self.frames[serial_number] = None
            self.frames[self.camera_names[len(self.pipelines) - 1]] = None
            self.timestamps[self.camera_names[len(self.pipelines) - 1]] = None
        # Start threads for capturing frames
        self.threads = []
        for i, pipeline in enumerate(self.pipelines):
            thread = threading.Thread(target=self._capture_frames, args=(self.camera_names[i], pipeline))
            thread.start()
            self.threads.append(thread)

    def _capture_frames(self, camera_name, pipeline):
        last_timestamp = time.time()
        while self.running:
            start_time = time.time()
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    logging.warning(f"No frames received from {camera_name}")
                    # print(f"No frames received from {camera_name}")
                    continue
                image = np.asanyarray(color_frame.get_data())
                current_timestamp = time.time()
                with self.lock:
                    self.frames[camera_name] = image
                    self.timestamps[camera_name] = current_timestamp
                    time_difference = current_timestamp - last_timestamp
                    last_timestamp = current_timestamp
                # logging.debug(
                #     f"{camera_name} updated at {current_timestamp}, time difference: {time_difference:.4f} seconds")
                # print(f"{camera_name} updated at {current_timestamp}, time difference: {time_difference:.5f} seconds")
            except RuntimeError as e:
                logging.error(f"Error capturing frames from {camera_name}: {e}")
                # print(f"Error capturing frames from {camera_name}: {e}")
                # time.sleep(0.05)
                # try:
                #     logging.info(f"Restarting camera pipeline for {camera_name}")
                #     pipeline.stop()
                #     pipeline = rs.pipeline()
                #     pipeline.start(config)
                #     self.pipelines.append((pipeline, config, serial_number))
                # except Exception as restart_error:
                #     logging.error(f"Failed to restart camera pipeline for {camera_name}: {restart_error}")
            except Exception as e:
                print(f"Unexpected error: {e}")
            elapsed_time = time.time() - start_time
            # print(f"{camera_name} elapsed time: {elapsed_time}")
            sleep_time = max(0, (0.03 - elapsed_time))
            time.sleep(sleep_time)

    def get_frames(self):
        with self.lock:
            return copy.deepcopy(self.frames)

    def stop(self):
        self.running = False
        for thread in self.threads:
            thread.join()
        for pipeline in self.pipelines:
            pipeline.stop()


if __name__ == '__main__':
    # camera_high, camera_left, camera_right = camera_init()
    # camera_high = camera_init()
    # print(camera_high)
    camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    serial_numbers = [
        'REPLACE_WITH_HIGH_CAMERA_SERIAL',
        'REPLACE_WITH_LEFT_WRIST_CAMERA_SERIAL',
        'REPLACE_WITH_RIGHT_WRIST_CAMERA_SERIAL',
    ]
    camera_reader = MultiCameraReader(serial_numbers)
    ckpt_dir = '/home/rm/rm_aloha_data_review/camera_test'
    time.sleep(2)
    images_list = []
    for i in range(500):
        # time.sleep(0.02)
        t0 = time.time()
        image_dict = camera_reader.get_frames()

        images_list.append(image_dict)
        t1 = time.time()
        print('time:', t1 - t0, " index: ", i)
        time.sleep(max(0.02, 0.02 - (t1 - t0)))
        # print(image_dict)
    camera_reader.stop()
    cv2.destroyAllWindows()
    data_dict = {}
    for cam_name in camera_names:
        data_dict[f'{cam_name}'] = []
    for i in images_list:
        for cam_name in camera_names:
            data_dict[f'{cam_name}'].append(i[cam_name])
    # 将图像数据转成视频
    for cam_name in camera_names:
        height, width, layers = data_dict[cam_name][0].shape

        # 为视频创建一个VideoWriter对象
        video_filename = os.path.join(ckpt_dir, "video_" + cam_name + ".avi")
        fource = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(video_filename, fource, 50, (width, height))
        for image in data_dict[cam_name]:
            # image = BGR_to_RGB(image)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            video.write(image)

        video.release()
        print(f"Video saved as {video_filename}")


    # # Display captured frames
    # for i in range(10):
    #     cv2.imshow(f"High {i}", images_list[i]['camera_high'])
    #     cv2.imshow(f"Left {i}", images_list[i]['camera_left'])
    #     cv2.imshow(f"Right {i}", images_list[i]['camera_right'])
    #     cv2.waitKey(0)
