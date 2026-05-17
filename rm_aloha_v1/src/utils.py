import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from scipy.signal import medfilt
import IPython
from src.constants import Remove_high

e = IPython.embed


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def apply_median_filter(data, kernel_size=5):
    # print("median_filter:", data.shape)
    filtered_data = np.copy(data)
    for i in range(data.shape[1]):
        filtered_data[:, i] = medfilt(data[:, i], kernel_size=kernel_size)
    return filtered_data


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, policy_class):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids  # 训练集的索引[1,2,3,....,20]
        self.dataset_path_list = dataset_path_list  # 展平的hdf5文件列表
        self.camera_names = camera_names  # ['top', 'left_wrist', 'right_wrist']
        self.norm_stats = norm_stats  # 数据集的统计信息
        self.episode_len = episode_len  # 每个数据集的长度[1200,1200,1200,....,1200]
        self.chunk_size = chunk_size  # Q 一个批次的大小 ？？
        self.cumulative_len = np.cumsum(self.episode_len)  # 累加每个数据集的长度
        print('max(episode_len)', max(episode_len))
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class  # 策略信息
        if self.policy_class == 'Diffusion':
            self.augment_images = True  # 如果是‘Diffusion' 数据增强
        else:
            self.augment_images = False  # 其他策略不启动数据增强
        self.transformations = None
        self.__getitem__(0)  # initialize self.is_sim and self.transformations
        self.is_sim = False

    # def __len__(self):
    #     return sum(self.episode_len)

    # 根据全局索引获取 具体数据的索引 和 数据的起始位置
    def _locate_transition(self, index):
        # 确保index在episode_len的范围内
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index)  # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        # 根据全局索引获取 具体数据的索引 和 数据的起始位置(当前这个episode内的数据位置)
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]  # 根据索引获取数据集的路径
        try:
            # print(dataset_path)
            with h5py.File(dataset_path, 'r') as root:
                try:  # some legacy data does not have this attribute
                    is_sim = root.attrs['sim']
                except:
                    is_sim = False
                compressed = root.attrs.get('compress', False)
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root['/action'][()], base_action], axis=-1)
                else:
                    action = root['/action'][()]
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)
                # at 中值滤波
                action = apply_median_filter(action)
                original_action_shape = action.shape  # 1200 * 14
                episode_len = original_action_shape[0]  # 1200
                # at 只获取start_ts时刻的观测数据 qpos qvel image 动作获取的更多
                qpos = root['/observations/qpos'][start_ts]
                # qvel = root['/observations/qvel'][start_ts]
                image_dict = dict()
                for cam_name in self.camera_names:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

                # 压缩恢复 at 有点问题，在压缩时做了数据的填充，但是在解压时没有做相应的处理 !!!
                if compressed:
                    for cam_name in image_dict.keys():
                        decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                        image_dict[cam_name] = np.array(decompressed_image)

                #去掉中间相机 将某个相机的图像全部填充为0
                if Remove_high:
                    image_dict['cam_high'] = np.zeros_like(image_dict['cam_high'])

                # 全面获取了当前这个episode的全部动作信息 需要切片
                if is_sim:
                    action = action[start_ts:]
                    action_len = episode_len - start_ts
                else:
                    # Q 为什么要减1 在获取的时候 不直接action = root['/action'][start_ts:] ????
                    action = action[max(0, start_ts - 1):]  # hack, to make timesteps more aligned
                    action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

            # at 将读取的动作数据和观测数据进行填充、转换和增强
            # self.is_sim = is_sim
            # 1200 * 14
            padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
            padded_action[:action_len] = action  # 前action_len个数据填充
            is_pad = np.zeros(self.max_episode_len)
            is_pad[action_len:] = 1  # 后面的数据填充为1

            # at chunk_size的作用是什么 是预测后k步动作嘛？？？
            padded_action = padded_action[:self.chunk_size]
            is_pad = is_pad[:self.chunk_size]

            # new axis for different cameras
            all_cam_images = []
            for cam_name in self.camera_names:
                all_cam_images.append(image_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=0)

            # construct observations
            image_data = torch.from_numpy(all_cam_images)
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()

            # 调整张量的维度 NCHW是PyTorch中图像的默认格式
            image_data = torch.einsum('k h w c -> k c h w', image_data)

            # augmentation
            if self.transformations is None:
                print('Initializing transformations')
                original_size = image_data.shape[2:]  # 获得图像的大小
                ratio = 0.95  # 比例
                # 随机裁减，调整大小到原始尺寸，旋转[-5,5]度，随机调整亮度、对比度、饱和度
                self.transformations = [
                    transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                    transforms.Resize(original_size, antialias=True),
                    transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)  # , hue=0.08)
                ]
            # 对图像进行增强
            if self.augment_images:
                for transform in self.transformations:
                    image_data = transform(image_data)

            # normalize image and change dtype to float
            image_data = image_data / 255.0

            if self.policy_class == 'Diffusion':
                # normalize to [-1, 1]
                action_data = ((action_data - self.norm_stats["action_min"]) / (
                        self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
            else:
                # normalize to mean 0 std 1
                action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        except:
            print(f'Error loading {dataset_path} in __getitem__')
            quit()

        # print(image_data.dtype, qpos_data.dtype, action_data.dtype, is_pad.dtype)
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_path_list):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            # 以读的方式打开hdf5文件
            with h5py.File(dataset_path, 'r') as root:
                # at 数据的保存形式 需要对应上
                # 读取pqos qvel action三类数据
                qpos = root['/observations/qpos'][()]
                # qvel = root['/observations/qvel'][()]
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    # 卷积平滑处理
                    base_action = preprocess_base_action(base_action)
                    # 拼接action和base_action
                    action = np.concatenate([root['/action'][()], base_action], axis=-1)
                else:
                    action = root['/action'][()]
                    # 生成N*2的0矩阵 代表base_action 为了凑齐维度 1200*14 我们只有机械臂的12(6*2)个维度
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        #  读出数据 qpos action
        # at 中值滤波
        action = apply_median_filter(action)

        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
    # print('all_qpos_data',all_qpos_data)
    # 沿着指定维度拼接 这里沿着第一个维度拼接
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # 取第一个维度的均值和方差 把防止限制在1e-2和无穷大之间 最后取最大值和最小值
    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps, "action_max": action_max.numpy() + eps,
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
             "example_qpos": qpos}  # Q 这个 example_qpos的作用是什么???

    # 返回state all_episode_len 这个记录的是每个qpos的长度
    return stats, all_episode_len


def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files


def BatchSampler(batch_size, episode_len_l, sample_weights):
    # 根据采样权重计算采样概率
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    # episode_len_l = [[1,2,3],[4,5,6],[7,8,9]] 三个数据集的长度
    # np.cumsum() 累加函数 [0,6,21,45] 代表每个数据集的长度
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            # np.random.choice 用于从指定的序列中随机选择一个元素
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            # np.random.randint 用于生成一个在指定范围内的随机整数
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch


def load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size,
              skip_mirrored_data=False, load_pretrain=False, policy_class=None, stats_dir_l=None, sample_weights=None,
              train_ratio=0.99):
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    print('dataset_dir_l', dataset_dir_l)
    # 二维数组 多个路径的训练数据集 [['/home/rm/rm_aloha/rm_aloha/rm_data/episode_6.hdf5',....],[],[]]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    print('dataset_path_list_list=========================', dataset_path_list_list)
    num_episodes_0 = len(dataset_path_list_list[0])
    # 嵌入列表展平成但一列表
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    # 获取每个路径的数据集长度 [20,xx,]
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    # 累加每个路径的数据集长度 ----> 总数 [20,xx,xx] 可以用来计数 切片
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # obtain train test split on dataset_dir_l[0]
    # 对给定数组或范围内的数值进行随机排列。 Q 生成一个打乱的数组 0~num_episodes_0
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    # 生成训练集和验证集 （打乱的数据集）
    train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]
    # 生成训练集和测试集的索引 train_episode_ids_0是一个打乱的索引 + num_episodes_cumsum[0]是一个偏移量也就是前总和的数量
    # [[1,2,3,][]]
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for
                                                   idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    # 按轴拼接到一起 [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20] 因为第一个按比例抽取了，所以第一个有缺失
    # 所有数据集的索引
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(
        f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')

    # obtain normalization stats for qpos and action
    # if load_pretrain:
    #     with open(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'dataset_stats.pkl'), 'rb') as f:
    #         norm_stats = pickle.load(f)
    #     print('Loaded pretrain dataset stats')
    # 获得了每个数据集的长度
    _, all_episode_len = get_norm_stats(dataset_path_list)
    # 根据训练集索引列表提取每个训练集样本的长度
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    print('train_episode_len_l', train_episode_len_l)
    # train_episode_len_l[[1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200]]
    # 根据验证集索引列表提取每个验证集样本的长度
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    # 将嵌套的列表展平为一个单一的列表 把嵌套的列表展平为一个单一的列表
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)

    # at stats_dir_l为None的时候 相当于读了两次数据集 一次获取了数据集的长度 一次获取了数据集的统计信息
    if stats_dir_l is None:  # stats_dir_l = None
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    norm_stats, _ = get_norm_stats(
        flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]))
    print(f'Norm stats from: {stats_dir_l}')

    # 批次采样 返回的是一个batch的索引 这里索引指的帧索引
    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

    # print(f'train_episode_len: {train_episode_len}, val_episode_len: {val_episode_len}, train_episode_ids: {train_episode_ids}, val_episode_ids: {val_episode_ids}')

    # construct dataset and dataloader Q 构造数据集和数据加载器
    # 单个加载
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len,
                                    chunk_size, policy_class)
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len,
                                  chunk_size, policy_class)
    train_num_workers = (8 if os.getlogin() == 'zfu' else 16) if train_dataset.augment_images else 2
    val_num_workers = 8 if train_dataset.augment_images else 2
    print(
        f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
    # AT 将数据集分割成多个小批次，同时提供对数据的并行加载和转换
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True,
                                  num_workers=train_num_workers, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True,
                                num_workers=val_num_workers, prefetch_factor=2)

    # at 数据集采集 并未使用到 速度 和 力
    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0  # 0.19
    v = base_action[..., 0]
    w = base_action[..., 1]
    base_action = base_action.copy()
    base_action[..., 0] = v - c * w
    return base_action


def smooth_base_action(base_action):
    # 对base_action的每一列进行卷积操作 5个一组进行平均 最后堆叠起来 转换为float32
    # 其中 mode='same'表示输出与输入的长度相同 采用还是0填充
    return np.stack([
        np.convolve(base_action[:, i], np.ones(5) / 5, mode='same') for i in range(base_action.shape[1])
    ], axis=-1).astype(np.float32)


def preprocess_base_action(base_action):
    # base_action = calibrate_linear_vel(base_action)
    base_action = smooth_base_action(base_action)

    return base_action


def postprocess_base_action(base_action):
    linear_vel, angular_vel = base_action
    linear_vel *= 1.0
    angular_vel *= 1.0
    # angular_vel = 0
    # if np.abs(linear_vel) < 0.05:
    #     linear_vel = 0
    return np.array([linear_vel, angular_vel])


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
