import os
import time

import h5py
import fnmatch
import numpy as np
import cv2
import matplotlib
import re
from scipy.signal import medfilt

print(matplotlib.get_backend())
matplotlib.use('Agg')
import matplotlib.pyplot as plt


pi = np.pi


def apply_median_filter(data, kernel_size=5):
    # print("median_filter:", data.shape)
    filtered_data = np.copy(data)
    for i in range(data.shape[1]):
        filtered_data[:, i] = medfilt(data[:, i], kernel_size=kernel_size)
    return filtered_data


class LowPassFilter:
    """
    dT: sec
    freq: Hz
    """

    def __init__(self, dT, freq):
        self.dT = dT
        self.damping = 0.75
        wn = 2 * pi * freq
        tmp1 = dT * dT * wn * wn
        tmp2 = 4 * self.damping * dT * wn
        a0 = tmp1 + tmp2 + 4.0
        a1 = 1
        a2 = (2 * tmp1 - 8.0) / a0
        a3 = (tmp1 - tmp2 + 4.0) / a0
        b0 = tmp1 / a0
        b1 = 2 * tmp1 / a0
        b2 = b0
        self.a = [a1, a2, a3]
        self.b = [b0, b1, b2]
        self.x = [0, 0]
        self.y = [0, 0]
        self.is_first_run = True

    def filter(self, input):
        if self.is_first_run:
            self.x = [input, input]
            self.y = [input, input]
            self.is_first_run = False

        output = self.b[0] * input + self.b[1] * self.x[0] + self.b[2] * self.x[1] - self.a[1] * self.y[0] - self.a[2] * \
                 self.y[1]
        self.y[1] = self.y[0]
        self.y[0] = output

        self.x[1] = self.x[0]
        self.x[0] = input

        return output


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


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def save_plot_1(ckpt_dir, obj, filter, name, index):
    state_dim = obj.shape[1]
    size = obj.shape[0]
    # 创建一个绘图窗口，设置大小
    # 15，20
    if filter is False:
        plt.figure(figsize=(15, 30))
        # 遍历每个状态维度并创建子图
        for i in range(state_dim):
            plt.subplot(state_dim, 1, i + 1)
            plt.plot(obj[:, i])
            if i < 6:
                plt.title(f'Joint_l_{name} {i + 1}')
            elif 6 < i < 13:
                plt.title(f'Joint_r_{name} {i + 1 - 7}')
            else:
                side = 'l' if i == 6 else 'r'
                plt.title(f'Grasper_{side}_{name}')
            # 设置 x 轴范围，确保显示所有样本
            # plt.xlim(0, size)
            # 添加图例以确认两条曲线都被绘制出来
            # plt.legend()
    else:
        dT = 0.02
        freq = 10  # 滤波频率为2Hz
        obj_filter = obj
        obj_filter = np.array(obj_filter)
        lpf_cur = LowPassFilter(dT, freq)
        time1 = time.time()  # 获取当前时间
        for i in range(obj_filter.shape[0]):
            obj_filter[i] = lpf_cur.filter(obj_filter[i].T).T
        print(obj_filter.shape)
        time2 = time.time()
        print('time:', time2 - time1)
        # 创建一个图形对象和多个子图
        fig, axs = plt.subplots(state_dim, 1, figsize=(15, 3 * state_dim))
        for i in range(state_dim):
            ax1 = axs[i]
            ax2 = ax1.twinx()
            # 绘制第一条曲线
            line1, = ax1.plot(obj[:, i], label='origin', color='blue')
            # 绘制第二条曲线
            line2, = ax2.plot(obj_filter[:, i], label='filter', color='orange')
            if i < 6:
                ax1.set_title(f'Joint_l_{name} {i + 1}')
            elif 6 < i < 13:
                ax1.set_title(f'Joint_r_{name} {i + 1 - 7}')
            else:
                side = 'l' if i == 6 else 'r'
                ax1.set_title(f'Grasper_{side}_{name}')
            # 设置 y 轴标签
            ax1.set_ylabel('origin')
            ax2.set_ylabel('filter')
            # 添加图例
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')

    # 调整布局以避免子图重叠
    plt.tight_layout()
    # 显示绘图
    plt.savefig(os.path.join(ckpt_dir, f'{name}_{index}.png'))
    plt.close()


def save_plot_2(ckpt_dir, obj, name, obj1, name1, index):
    state_dim = obj.shape[1]
    size = obj.shape[0]
    # 创建一个绘图窗口，设置大小
    # 15，20
    fig, axs = plt.subplots(state_dim, 1, figsize=(15, 3 * state_dim))
    for i in range(state_dim):
        ax1 = axs[i]
        ax2 = ax1.twinx()
        # 绘制第一条曲线
        line1, = ax1.plot(obj[:, i], label=f'{name}', color='blue')
        # 绘制第二条曲线
        line2, = ax2.plot(obj1[:, i], label=f'{name1}', color='orange')
        if i < 6:
            ax1.set_title(f'Joint_l_ {i + 1}')
        elif 6 < i < 13:
            ax1.set_title(f'Joint_r_{i + 1 - 7}')
        else:
            side = 'l' if i == 6 else 'r'
            ax1.set_title(f'Grasper_{side}_{name}')
        # 设置 y 轴标签
        ax1.set_ylabel(f'{name}')
        ax2.set_ylabel(f'{name1}')
        # 添加图例
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
    # 调整布局以避免子图重叠
    plt.tight_layout()
    # 显示绘图
    plt.savefig(os.path.join(ckpt_dir, f'{name}_and_{name1}_{index}.png'))
    plt.close()


def BGR_to_RGB(cvimg):
    pilimg = cvimg.copy()
    pilimg[:, :, 0] = cvimg[:, :, 2]
    pilimg[:, :, 2] = cvimg[:, :, 0]
    return pilimg


def read_hdf5(ckpt_dir, hdf5_file, camera_names):
    for file_path in hdf5_file:
        # 使用正则表达式匹配数字
        match = re.search(r'episode_(\d+)', file_path)
        if not match:
            print(f"Error: episode number not found in {file_path}")
            continue
        index = int(match.group(1))
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        if os.path.isfile(os.path.join(ckpt_dir, f'action_{index}.png')):
            continue
        try:
            # dataset_path_list[index]
            # '/home/rm/rm_aloha/rm_aloha/rm_data/episode_19.hdf5'
            with h5py.File(file_path, 'r') as root:
                compressed = root.attrs.get('compress', False)
                qpos = root['/observations/qpos']
                # qvel = root['/observations/qvel']
                # effort = root['/observations/effort']
                action = root['/action']

                compress_len = root['/compress_len']

                # print(effort.shape)
                # effort = np.array(effort)
                # effort = effort.T
                # print(effort.shape)
                # time1 = time.time()  # 获取当前时间
                # for i in range(effort.shape[0]):
                #     effort[i] = lpf_cur.filter(effort[i].T).T
                # print(effort.shape)
                # time2 = time.time()
                # print('time:', time2 - time1)
                save_plot_1(ckpt_dir, qpos, False, 'qpos', index)
                # save_plot(ckpt_dir, qvel, False, 'qvel', index)
                # save_plot(ckpt_dir, effort, True, 'effort', index)
                # at action的含义是什么？ ===> action采集的是主动关节的位置信息，也就是我们控制的机械臂
                save_plot_1(ckpt_dir, action, False, 'action', index)
                filtered_action = apply_median_filter(action)

                print(filtered_action.shape)
                save_plot_1(ckpt_dir, filtered_action, False, 'action_filter', index)
                save_plot_2(ckpt_dir, action, 'action', qpos, 'qpos', index)
                image_dict = dict()
                for cam_name in camera_names:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}']

                if compressed:
                    for cam_name in camera_names:
                        padded_compressed_image_list = image_dict[cam_name]
                        # print('image_list:', padded_compressed_image_list.shape)
                        decoded_image_list = []

                        for i in range(len(padded_compressed_image_list)):
                            # print(i)
                            valid_compressed_image = []
                            padded_compressed_image = padded_compressed_image_list[i]
                            image_len = round(compress_len[camera_names.index(cam_name)][i])
                            # print(image_len)
                            if len(padded_compressed_image.shape) != 1:
                                print(f"Error: padded_compressed_image for {cam_name}, image {i} is not 1D")
                                continue
                            if image_len <= len(padded_compressed_image):
                                valid_compressed_image = padded_compressed_image[:image_len]
                                print('valid_compressed_image:', valid_compressed_image.shape)
                            # 从字节字符串中解码图像
                            decompressed_image = cv2.imdecode(valid_compressed_image, cv2.IMREAD_COLOR)
                            # print(decompressed_image.shape)
                            decoded_image_list.append(decompressed_image)

                        image_dict[cam_name] = decoded_image_list
                        # 打印读取到的图像数据

                # 将图像数据转成视频
                for cam_name in camera_names:
                    height, width, layers = image_dict[cam_name][0].shape

                    # 为视频创建一个VideoWriter对象
                    video_filename = os.path.join(ckpt_dir, "video_" + cam_name + "_" + str(index) + ".avi")
                    fource = cv2.VideoWriter_fourcc(*'XVID')
                    video = cv2.VideoWriter(video_filename, fource, 50, (width, height))
                    for image in image_dict[cam_name]:
                        # image = BGR_to_RGB(image)
                        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        video.write(image)

                    video.release()
                    print(f"Video saved as {video_filename}")

        except:
            quit()


def main():
    # dataset_dir_l = './aloha_data/rm_data_801'
    dataset_dir_l = '/home/nuaacmee/rm_aloha_v1/aloha_data'
    ckpt_dir = '/home/nuaacmee/rm_aloha_v1/aloha_data_visualize'
    camera_names = ['cam_left_wrist', 'cam_right_wrist']
    # print(camera_names.index('cam_high'))
    skip_mirrored_data = True
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    # print('dataset_path_list_list=========================', dataset_path_list_list)
    dataset_path_list = flatten_list(dataset_path_list_list)
    # print('dataset_path_list_list=========================', dataset_path_list)
    datalen = len(dataset_path_list)
    print(datalen)
    # dataset_path_list = ['/home/rm/rm_aloha_imdata/rm_data_725_1/episode_83.hdf5']
    read_hdf5(ckpt_dir, dataset_path_list, camera_names)


if __name__ == '__main__':
    main()
