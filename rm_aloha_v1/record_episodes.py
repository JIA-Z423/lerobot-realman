import os
import time
import h5py
import argparse
import numpy as np
from tqdm import tqdm
import cv2
from utils.get_master_joints import get_master_act, arm_conf
from utils.robot_utils import Recorder, ImageRecorder
from utils.real_env import make_real_env
import IPython

e = IPython.embed


def capture_one_episode(DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite, l_ser, r_ser, byte_send,
                        arm_left_ip, arm_right_ip, serial_numbers):
    print(f'Dataset name: {dataset_name}')

    env = make_real_env(arm_left_ip, arm_right_ip, serial_numbers)
    print("=====================================Start!=============================================")
    time.sleep(0.5)

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    # Data collection
    ts = env.reset(fake=True)
    available_camera_names = list(ts.observation.get('images', {}).keys())
    missing_camera_names = [cam for cam in camera_names if cam not in available_camera_names]
    if missing_camera_names:
        print(f'[WARN] Requested cameras not found: {missing_camera_names}')
        print(f'[WARN] Available cameras from env: {available_camera_names}')
        camera_names = [cam for cam in camera_names if cam in available_camera_names]
        if not camera_names:
            raise RuntimeError(
                f'None of requested cameras are available. '
                f'Requested={missing_camera_names}, Available={available_camera_names}'
            )
        print(f'[INFO] Falling back to available cameras: {camera_names}')
    timesteps = [ts]
    actions = []
    time0 = time.time()
    actual_dt_history = []

    for t in tqdm(range(max_timesteps)):
        t0 = time.time()  # 获取时间
        action = get_master_act(l_ser, r_ser, byte_send)  # 获取主动关节动作
        t1 = time.time()
        ts = env.step()  # 返回环境信息 机械臂的位置，速度，力信息，图像信息
        t2 = time.time()
        timesteps.append(ts)
        actions.append(action)
        actual_dt_history.append([t0, t1, t2])
        time.sleep(max(0, DT - (time.time() - t0)))  # 确保每次间隔DT时间 0.02s
    print(f'Avg fps: {max_timesteps / (time.time() - time0)}')
    env.camera_reader.stop()

    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 30:
        print(f'\n\nfreq_mean is {freq_mean}, lower than 30, re-collecting... \n\n\n\n')

    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
        - cam_left_wrist    (480, 640, 3) 'uint8'
        - cam_right_wrist   (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'
    - effort                (14,)         'float64'
        
    action                  (14,)         'float64'
    base_action             (2,)          'float64'
    """

    data_dict = {
        '/observations/qpos': [],
        # '/observations/qvel': [],
        # '/observations/effort': [],
        '/action': [],
        # '/base_action': [],
        # '/base_action_t265': [],
    }
    # 初始化相机信息
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # 保存信息至 data_dict
    # qpos姿态信息 qvel速度信息 effort力信息？ action 动作信息 images 图片信息
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        # data_dict['/observations/qvel'].append(ts.observation['qvel'])
        # data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/action'].append(action)
        # data_dict['/base_action'].append(ts.observation['base_vel'])
        # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
        image_obs = ts.observation.get('images', {})
        for cam_name in camera_names:
            if cam_name not in image_obs:
                raise KeyError(
                    f"Camera '{cam_name}' is missing at runtime. "
                    f"Available keys: {list(image_obs.keys())}"
                )
            data_dict[f'/observations/images/{cam_name}'].append(image_obs[cam_name])

    # 压缩
    COMPRESS = True
    if COMPRESS:
        # JPEG compression
        t0 = time.time()
        # jpg的格式 指定编码的参数
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # tried as low as 20, seems fine
        compressed_len = []

        # 对图像数据重新处理 compressed_list[]  compressed_len[][]
        for cam_name in camera_names:
            image_list = data_dict[f'/observations/images/{cam_name}']
            compressed_list = []
            # 二维数组
            compressed_len.append([])
            # 对每张图片进行重新编码成.jpg形式 然后覆盖掉之前的数据
            for image in image_list:
                # 编码后的图像数据 编码后是一维数组
                # print('IAMGE.SIZE', image.shape)
                result, encoded_image = cv2.imencode('.jpg', image,
                                                     encode_param)  # 0.02 sec # cv2.imdecode(encoded_image, 1)
                compressed_list.append(encoded_image)
                # 记录每张图片的长度
                compressed_len[-1].append(len(encoded_image))
            data_dict[f'/observations/images/{cam_name}'] = compressed_list
        print(f'compression: {time.time() - t0:.2f}s')

        # pad so it has same length
        # 填充图片到一致
        t0 = time.time()
        compressed_len = np.array(compressed_len)
        print('compressed_len', compressed_len.shape)
        # 找大最大的图片/长度
        padded_size = compressed_len.max()
        print('padded_size', padded_size)
        # 图片填充，使得大小一致
        for cam_name in camera_names:
            compressed_image_list = data_dict[f'/observations/images/{cam_name}']
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                # 生成一个全0的数组 并且是最长的图片长度
                padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                image_len = len(compressed_image)
                # 保持原有长度数据 at 只能知道图片最大长度，并不确定哪些图片进行了填充，给解码带来了困难？？？
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
        print(f'padding: {time.time() - t0:.2f}s')

    # HDF5 每次就保存一个
    t0 = time.time()
    # max_timesteps = max_timesteps+1
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = COMPRESS
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            if COMPRESS:
                _ = image.create_dataset(cam_name, (max_timesteps, padded_size), dtype='uint8',
                                         chunks=(1, padded_size), )
            else:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
        # observations 环境数据大小
        _ = obs.create_dataset('qpos', (max_timesteps, 14))
        # _ = obs.create_dataset('qvel', (max_timesteps, 14))
        # _ = obs.create_dataset('effort', (max_timesteps, 14))
        # 动作信息
        _ = root.create_dataset('action', (max_timesteps, 14))
        # _ = root.create_dataset('base_action', (max_timesteps, 2))
        # _ = root.create_dataset('base_action_t265', (max_timesteps, 2))
        # print("data_dict", data_dict.items())
        for name, array in data_dict.items():
            # print("name", name, "array", array)
            root[name][...] = array

        if COMPRESS:
            _ = root.create_dataset('compress_len', (len(camera_names), max_timesteps))
            root['/compress_len'][...] = compressed_len

    print(f'Saving: {time.time() - t0:.1f} secs')

    return True


def main(args):

    dataset_dir = '/home/nuaacmee/rm_aloha_v1/aloha_data'
    max_timesteps = 400  # 最大时间步
    DT = 0.02  # 时间间隔
    camera_names = ['cam_left_wrist', 'cam_right_wrist']  # 相机名称
    args['episode_idx'] = None
    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True  # 是否覆盖

    dataset_name = f'episode_{episode_idx}'
    print(dataset_name + '\n')

    l_ser, r_ser, byte_send = arm_conf()  # 初始化主动臂
    arm_left_ip = "REPLACE_WITH_LEFT_REALMAN_ARM_HOST"
    arm_right_ip = "REPLACE_WITH_RIGHT_REALMAN_ARM_HOST"
    serial_numbers = ['REPLACE_WITH_LEFT_WRIST_CAMERA_SERIAL', 'REPLACE_WITH_RIGHT_WRIST_CAMERA_SERIAL']
    if len(serial_numbers) == 2 and camera_names == ['cam_left_wrist', 'cam_right_wrist']:
        print('[WARN] Only 2 camera serials are configured, but default mapping is '
              '[cam_high, cam_left_wrist, cam_right_wrist].')
        print('[WARN] With 2 serials, camera keys are likely [cam_high, cam_left_wrist], '
              'so cam_right_wrist may be missing.')
    
    # while True:
    #     # 采集数据
    #     is_healthy = capture_one_episode(DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite,
    #                                      l_ser, r_ser, byte_send, arm_left_ip, arm_right_ip, serial_numbers)
    #     if is_healthy:
    #         break
    num_episodes = 3
    for i in range(num_episodes):
        episode_idx = get_auto_index(dataset_dir)
        dataset_name = f'episode_{episode_idx}'
        input(f"\n {dataset_name}，press enter to start") 

        while True:
             is_healthy = capture_one_episode(DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite,
                                              l_ser, r_ser, byte_send, arm_left_ip, arm_right_ip, serial_numbers)
             if is_healthy:
                 print(f"{dataset_name} capture completely ") 
                 break
             else:
                 print("Recapture this episode")
        
        
    


# 获取当前的索引
def get_auto_index(dataset_dir, dataset_name_prefix='', data_suffix='hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx + 1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(
        f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}')
    return freq_mean


def debug():
    print(f'====== Debug mode ======')
    recorder = Recorder('right', is_debug=True)
    image_recorder = ImageRecorder(init_node=False, is_debug=True)
    while True:
        time.sleep(1)
        recorder.print_diagnostics()
        image_recorder.print_diagnostics()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    # parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    main(vars(parser.parse_args()))  # TODO
    # debug()
