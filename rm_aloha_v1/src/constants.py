import pathlib
import os

DATA_DIR = os.path.expanduser('/home/rm/rm_aloha_imdata/')
Remove_high = False

TASK_CONFIGS = {
    # 第一次成功抓取的数据集
    # 特点：机械臂固定初始位姿，并且物品在视野内
    'rm_data_726_1': {
        'dataset_dir': DATA_DIR + '/rm_data_726_1',
        'episode_len': 50,
        'train_ratio': 0.9,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    # 单臂抓取数据集
    'rm_data_801': {
        'dataset_dir': DATA_DIR + '/rm_data_801',
        'episode_len': 50,
        'train_ratio': 0.9,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    # 特点：机械臂固定位姿 物品不在视野抓取
    'rm_data_805': {
        'dataset_dir': DATA_DIR + '/rm_data_805',
        'episode_len': 50,
        'train_ratio': 0.9,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    # 协同训练：rm_data_805 + rm_data_726_1 总体就是在或不在视野内抓取
    'rm_data_805_1': {
        'dataset_dir': [
            DATA_DIR + '/rm_data_805',
            DATA_DIR + '/rm_data_726_1',
        ],
        'episode_len': 50,
        'train_ratio': 0.9,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    # 固定物品 不同初始化位置 抓取
    # train_8_6   ---- 固定物品 不同初始化位置 抓取
    # train_8_6_1 ---- 去掉中间摄像头 / 数据集用7_26
    'rm_data_806': {
        'dataset_dir': DATA_DIR + '/rm_data_806',
        'episode_len': 50,
        'train_ratio': 0.9,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    # 有中间相机 旋转找目标 固定初始位置 目标有旋转  可以成功
    # -1 为不使用中间相机  成功率很低
    'rm_data_807': {
        'dataset_dir': DATA_DIR + '/rm_data_807',
        'episode_len': 50,
        'train_ratio': 0.9,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_mobile_wipe_wine_cotrain': {
        'dataset_dir': [
            DATA_DIR + '/aloha_mobile_wipe_wine',
            DATA_DIR + '/aloha_compressed_dataset',
        ],  # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_mobile_wipe_wine',
        ],
        'sample_weights': [5, 5],
        'train_ratio': 0.9,  # ratio of train data from the first dataset_dir
        'episode_len': 1300,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_mobile_wipe_wine_2_cotrain': {
        'dataset_dir': [
            DATA_DIR + '/aloha_mobile_wipe_wine_2',
            DATA_DIR + '/aloha_compressed_dataset',
        ],  # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_mobile_wipe_wine_2',
        ],
        'sample_weights': [5, 5],
        'train_ratio': 0.9,  # ratio of train data from the first dataset_dir
        'episode_len': 1300,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    # cabinet
    'aloha_mobile_cabinet': {
        'dataset_dir': [
            DATA_DIR + '/aloha_mobile_cabinet',
            DATA_DIR + '/aloha_mobile_cabinet_handles',  # 200
            DATA_DIR + '/aloha_mobile_cabinet_grasp_pots',  # 200
        ],  # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_mobile_cabinet',
        ],
        'sample_weights': [6, 1, 1],
        'train_ratio': 0.99,  # ratio of train data from the first dataset_dir
        'episode_len': 1500,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_mobile_cabinet_cotrain': {
        'dataset_dir': [
            DATA_DIR + '/aloha_mobile_cabinet',
            DATA_DIR + '/aloha_mobile_cabinet_handles',
            DATA_DIR + '/aloha_mobile_cabinet_grasp_pots',
            DATA_DIR + '/aloha_compressed_dataset',
        ],  # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_mobile_cabinet',
        ],
        'sample_weights': [6, 1, 1, 2],
        'train_ratio': 0.99,  # ratio of train data from the first dataset_dir
        'episode_len': 1500,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    'aloha_mobile_elevator_truncated': {
        'dataset_dir': [
            DATA_DIR + '/aloha_mobile_elevator_truncated',
            DATA_DIR + '/aloha_mobile_elevator_2',  # 1200
            DATA_DIR + '/aloha_mobile_elevator_button',  # 800
        ],  # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_mobile_elevator_truncated',
            DATA_DIR + '/aloha_mobile_elevator_2',
        ],
        'sample_weights': [3, 3, 2],
        'train_ratio': 0.99,  # ratio of train data from the first dataset_dir
        'episode_len': 2250,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_mobile_elevator_truncated_cotrain': {
        'dataset_dir': [
            DATA_DIR + '/aloha_mobile_elevator_truncated',
            DATA_DIR + '/aloha_mobile_elevator_2',
            DATA_DIR + '/aloha_mobile_elevator_button',
            DATA_DIR + '/aloha_compressed_dataset',
        ],  # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_mobile_elevator_truncated',
            DATA_DIR + '/aloha_mobile_elevator_2',
        ],
        'sample_weights': [3, 3, 2, 1],
        'train_ratio': 0.99,  # ratio of train data from the first dataset_dir
        'episode_len': 2250,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

}

### ALOHA fixed constants
DT = 0.02
FPS = 50