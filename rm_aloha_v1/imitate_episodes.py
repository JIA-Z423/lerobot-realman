import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange
import wandb
import time
from torchvision import transforms

from src.constants import FPS
from src.utils import load_data  # data functions
from src.utils import sample_box_pose, sample_insertion_pose
from src.utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, \
    postprocess_base_action  # helper functions
from network.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
# from visualize_episodes import save_videos
from RM_robotic.robotic_arm_package.robotic_arm import *

# from detr.models.latent_model import Latent_Model_Transformer
from src.sim_env import BOX_POSE

# import IPython
# e = IPython.embed

DROPOUT_RATE = 0.1

import torch.nn as nn
from torch.nn import functional as F
import torch


class Causal_Transformer_Block(nn.Module):
    def __init__(self, seq_len, latent_dim, num_head) -> None:
        super().__init__()
        self.num_head = num_head
        self.latent_dim = latent_dim
        self.ln_1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_head, dropout=DROPOUT_RATE, batch_first=True)
        self.ln_2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(DROPOUT_RATE),
        )

        # self.register_buffer("attn_mask", torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool())

    def forward(self, x):
        attn_mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool), diagonal=1)
        x = self.ln_1(x)
        x = x + self.attn(x, x, x, attn_mask=attn_mask)[0]
        x = self.ln_2(x)
        x = x + self.mlp(x)

        return x


class Latent_Model_Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, latent_dim=256, num_head=8, num_layer=3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.num_head = num_head
        self.num_layer = num_layer
        self.input_layer = nn.Linear(input_dim, latent_dim)
        self.weight_pos_embed = nn.Embedding(seq_len, latent_dim)
        self.attention_blocks = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            *[Causal_Transformer_Block(seq_len, latent_dim, num_head) for _ in range(num_layer)],
            nn.LayerNorm(latent_dim)
        )
        self.output_layer = nn.Linear(latent_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = x + self.weight_pos_embed(torch.arange(x.shape[1], device=x.device))
        x = self.attention_blocks(x)
        logits = self.output_layer(x)

        return logits

    @torch.no_grad()
    def generate(self, n, temperature=0.1, x=None):
        if x is None:
            x = torch.zeros((n, 1, self.input_dim), device=self.weight_pos_embed.weight.device)
        for i in range(self.seq_len):
            logits = self.forward(x)[:, -1]
            probs = torch.softmax(logits / temperature, dim=-1)
            samples = torch.multinomial(probs, num_samples=1)[..., 0]
            samples_one_hot = F.one_hot(samples.long(), num_classes=self.output_dim).float()
            x = torch.cat([x, samples_one_hot[:, None, :]], dim=1)

        return x[:, 1:, :]


def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx + 1):
        if not os.path.isfile(os.path.join(dataset_dir, f'qpos_{i}.npy')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def main(args, arm_left, arm_right, serial_numbers):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_steps = args['num_steps']
    eval_every = args['eval_every']
    validate_every = args['validate_every']
    save_every = args['save_every']
    resume_ckpt_path = args['resume_ckpt_path']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    print(task_name)

    if is_sim or task_name == 'all':
        from src.sim_constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        # print('is_sim',is_sim)
        # print('task_name',task_name)
        from src.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    # print('task_config-----------------',task_config)
    dataset_dir = task_config['dataset_dir']
    # print('dataset_dir',dataset_dir)
    # num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 0.99)
    name_filter = task_config.get('name_filter', lambda n: True)

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'  # backbone使用的resnet18
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'vq': args['use_vq'],
                         'vq_class': args['vq_class'],
                         'vq_dim': args['vq_dim'],
                         'action_dim': 16,
                         'no_encoder': args['no_encoder'],
                         }
    elif policy_class == 'Diffusion':

        policy_config = {'lr': args['lr'],
                         'camera_names': camera_names,
                         'action_dim': 16,
                         'observation_horizon': 1,
                         'action_horizon': 8,
                         'prediction_horizon': args['chunk_size'],
                         'num_queries': args['chunk_size'],
                         'num_inference_timesteps': 10,
                         'ema_power': 0.75,
                         'vq': False,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone': backbone, 'num_queries': 1,
                         'camera_names': camera_names, }
    else:
        raise NotImplementedError

    actuator_config = {
        'actuator_network_dir': args['actuator_network_dir'],
        'history_len': args['history_len'],
        'future_len': args['future_len'],
        'prediction_len': args['prediction_len'],
    }

    config = {
        'num_steps': num_steps,
        'eval_every': eval_every,
        'validate_every': validate_every,
        'save_every': save_every,
        'ckpt_dir': ckpt_dir,
        'resume_ckpt_path': resume_ckpt_path,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'load_pretrain': args['load_pretrain'],
        'actuator_config': actuator_config,
    }

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    config_path = os.path.join(ckpt_dir, 'config.pkl')
    expr_name = ckpt_dir.split('/')[-1]
    if not is_eval:
        # wandb.init(project="mobile-aloha2", reinit=True, entity="mobile-aloha2", name=expr_name)
        wandb.init(project="mobile-aloha2", reinit=True, name=expr_name)
        wandb.config.update(config)
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    #  is_eval = False
    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        # ckpt_names = [f'policy_step_10000_seed_0.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, arm_left, arm_right, serial_numbers,
                                               save_episode=True, num_rollouts=10)
            # wandb.log({'success_rate': success_rate, 'avg_return': avg_return})
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()
    print('dataset_dir-------------------------------', dataset_dir)
    # 数据加载器 全局状态
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, name_filter, camera_names, batch_size_train,
                                                           batch_size_val, args['chunk_size'],
                                                           args['skip_mirrored_data'], config['load_pretrain'],
                                                           policy_class, stats_dir_l=stats_dir,
                                                           sample_weights=sample_weights, train_ratio=train_ratio)
    print('train_dataloader====================', train_dataloader)
    # save dataset stats
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config, arm_left, arm_right, serial_numbers)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')
    wandb.finish()


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'Diffusion':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        # 3*640*480
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    # 沿着新的维度将输入张量序列堆叠起来
    curr_image = np.stack(curr_images, axis=0)
    # unsqueeze()函数的作用是在指定的位置增加一个维度
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)  # 1*9*640*480

    if rand_crop_resize:
        print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)

    return curr_image


def eval_bc(config, ckpt_name, arm_left_ip, arm_right_ip, serial_numbers_n, save_episode=True, num_rollouts=1, step=1):
    set_seed(1000)  # 设置随机种子
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']  # 仿真与否
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']  # Q 这个参数的含义是什么？
    onscreen_cam = 'angle'
    vq = config['policy_config']['vq']
    actuator_config = config['actuator_config']
    use_actuator_net = actuator_config['actuator_network_dir'] is not None

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(loading_status)  # <All keys matched successfully>
    policy.cuda()
    policy.eval()  # 评估模式
    if vq:  # vq 默认是false
        vq_dim = config['policy_config']['vq_dim']
        vq_class = config['policy_config']['vq_class']
        latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
        # latent_model_ckpt_path = os.path.join(ckpt_dir, 'latent_model_last.ckpt')
        latent_model_ckpt_path = os.path.join(ckpt_dir, 'latent_model_best.ckpt')
        # latent_model_ckpt_path = os.path.join(ckpt_dir, 'policy_step_10000_seed_0.ckpt')
        latent_model.deserialize(torch.load(latent_model_ckpt_path))  # 加载模型参数
        latent_model.eval()
        latent_model.cuda()
        print(f'Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}')
    else:
        print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    # if use_actuator_net:
    #     prediction_len = actuator_config['prediction_len']
    #     future_len = actuator_config['future_len']
    #     history_len = actuator_config['history_len']
    #     actuator_network_dir = actuator_config['actuator_network_dir']

    #     from act.train_actuator_network import ActuatorNetwork
    #     actuator_network = ActuatorNetwork(prediction_len)
    #     actuator_network_path = os.path.join(actuator_network_dir, 'actuator_net_last.ckpt')
    #     loading_status = actuator_network.load_state_dict(torch.load(actuator_network_path))
    #     actuator_network.eval()
    #     actuator_network.cuda()
    #     print(f'Loaded actuator network from: {actuator_network_path}, {loading_status}')

    #     actuator_stats_path  = os.path.join(actuator_network_dir, 'actuator_net_stats.pkl')
    #     with open(actuator_stats_path, 'rb') as f:
    #         actuator_stats = pickle.load(f)

    #     actuator_unnorm = lambda x: x * actuator_stats['commanded_speed_std'] + actuator_stats['commanded_speed_std']
    #     actuator_norm = lambda x: (x - actuator_stats['observed_speed_mean']) / actuator_stats['observed_speed_mean']
    #     def collect_base_action(all_actions, norm_episode_all_base_actions):
    #         post_processed_actions = post_process(all_actions.squeeze(0).cpu().numpy())
    #         norm_episode_all_base_actions += actuator_norm(post_processed_actions[:, -2:]).tolist()

    # 定义一个匿名函数 使得数据s_qpos的均值为0 方差为1
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    if policy_class == 'Diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from utils.real_env import make_real_env  # requires aloha
        env = make_real_env(arm_left_ip, arm_right_ip, serial_numbers_n)
        env_max_reward = 0
    else:
        from src.sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    print("query_frequency: ", query_frequency)
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
    if real_robot:
        BASE_DELAY = 13  # 最高可以增加到20
        query_frequency -= BASE_DELAY
    print("temporal_agg:", temporal_agg, "  query_frequency:", query_frequency)

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset

        ts = env.reset()  # 重置环境

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, 16]).cuda()

        # qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        qpos_history_raw = np.zeros((max_timesteps, state_dim))
        image_list = []  # 图片
        qpos_list = []  # 位置
        target_qpos_list = []  # 目标位置
        rewards = []  # 奖励
        # if use_actuator_net:
        #     norm_episode_all_base_actions = [actuator_norm(np.zeros(history_len, 2)).tolist()]
        # 推理（inference）过程中禁用梯度计算，以加速推理过程并节省内存。
        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS
            culmulated_delay = 0
            print("max_timesteps", max_timesteps, " DT: ", DT)
            for t in range(max_timesteps):
                time1 = time.time()
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                time2 = time.time()
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos_history_raw[t] = qpos_numpy
                # 标准化操作 归一化
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)  # 1*1*14
                # qpos_history[:, t] = qpos
                if t % query_frequency == 0:
                    curr_image = get_image(ts, camera_names, rand_crop_resize=(config['policy_class'] == 'Diffusion'))
                    # print('cur_image: ', curr_image)
                # print('get image: ', time.time() - time2)

                if t == 0:
                    # warm up
                    for _ in range(10):
                        policy(qpos, curr_image)  # Q 执行10次前向传播的目的是什么？？
                    print('network warm up done')
                    time1 = time.time()

                ### query policy
                time3 = time.time()
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        print("vq: ", vq)
                        if vq:
                            if rollout_id == 0:
                                for _ in range(10):
                                    vq_sample = latent_model.generate(1, temperature=1, x=None)
                                    print(torch.nonzero(vq_sample[0])[:, 1].cpu().numpy())
                            vq_sample = latent_model.generate(1, temperature=1, x=None)
                            all_actions = policy(qpos, curr_image, vq_sample=vq_sample)
                        else:
                            # 输出动作
                            all_actions = policy(qpos, curr_image)
                            print("all_actions: ", all_actions.shape)
                        # if use_actuator_net:
                        #     collect_base_action(all_actions, norm_episode_all_base_actions)
                        if real_robot:
                            # action 维度 bitch x num_step x feature
                            # 第一个维度全取 第二个维度 取到倒数13之前 第三个维度取到 倒数第二之前 并且按照最后一个维度拼接
                            # Q 这里拼接的目的是什么？？？
                            all_actions = torch.cat(
                                [all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)
                    if temporal_agg:
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        # t从 0 开始
                        raw_action = all_actions[:, t % query_frequency]
                        print("raw_action:", raw_action)
                        # if t % query_frequency == query_frequency - 1:
                        #     # zero out base actions to avoid overshooting
                        #     raw_action[0, -2:] = 0
                elif config['policy_class'] == "Diffusion":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                        # if use_actuator_net:
                        #     collect_base_action(all_actions, norm_episode_all_base_actions)
                        if real_robot:
                            all_actions = torch.cat(
                                [all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)
                    raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                    all_actions = raw_action.unsqueeze(0)
                    # if use_actuator_net:
                    #     collect_base_action(all_actions, norm_episode_all_base_actions)
                else:
                    raise NotImplementedError
                # print('query policy: ', time.time() - time3)

                ### post-process actions
                time4 = time.time()
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action[:-2]
                print("target_qpos: ", target_qpos)

                # if use_actuator_net:
                #     assert(not temporal_agg)
                #     if t % prediction_len == 0:
                #         offset_start_ts = t + history_len
                #         actuator_net_in = np.array(norm_episode_all_base_actions[offset_start_ts - history_len: offset_start_ts + future_len])
                #         actuator_net_in = torch.from_numpy(actuator_net_in).float().unsqueeze(dim=0).cuda()
                #         pred = actuator_network(actuator_net_in)
                #         base_action_chunk = actuator_unnorm(pred.detach().cpu().numpy()[0])
                #     base_action = base_action_chunk[t % prediction_len]
                # else:
                # base_action = action[-2:]
                # base_action = calibrate_linear_vel(base_action, c=0.19)
                # base_action = postprocess_base_action(base_action)
                # print('post process: ', time.time() - time4)

                ### step the environment
                time5 = time.time()
                if real_robot:
                    # Movej_Cmd:0 * 2 + Set_Gripper_Position:0 * 2
                    ts = env.step_rm(target_qpos, get_obs=True)
                else:
                    ts = env.step_rm(target_qpos)
                # print('step env: ', time.time() - time5)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
                duration = time.time() - time1
                sleep_time = max(0, DT - duration)
                time.sleep(sleep_time)
                # time.sleep(max(0, DT - duration - culmulated_delay))
                if duration >= DT:
                    culmulated_delay += (duration - DT)
                    print(
                        f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {DT} s, culmulated delay: {culmulated_delay:.3f} s')
                # else:
                #     culmulated_delay = max(0, culmulated_delay - (DT - duration))

            print(f'Avg fps {max_timesteps / (time.time() - time0)}')
            plt.close()
        if real_robot:
            # at 需要修改因为就是释放夹爪
            # env.set_gripper_release()
            # env.set_gripper_pose(1.0, 1.0)  # open Q ???
            # save qpos_history_raw
            log_id = get_auto_index(ckpt_dir)
            np.save(os.path.join(ckpt_dir, f'qpos_{log_id}.npy'), qpos_history_raw)
            plt.figure(figsize=(10, 20))
            # plot qpos_history_raw for each qpos dim using subplots
            for i in range(state_dim):
                plt.subplot(state_dim, 1, i + 1)
                plt.plot(qpos_history_raw[:, i])
                # remove x axis
                if i != state_dim - 1:
                    plt.xticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, f'qpos_{log_id}.png'))
            plt.close()

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward == env_max_reward}')

        # if save_episode:
        #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    env.camera_reader.stop()
    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate * 100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)  # TODO remove None


def train_bc(train_dataloader, val_dataloader, config, arm_left, arm_right, serial_numbers):
    num_steps = config['num_steps']  # 训练步数
    ckpt_dir = config['ckpt_dir']  # 检查点目录
    seed = config['seed']  # 随机种子
    policy_class = config['policy_class']  # 策略选择
    policy_config = config['policy_config']  # 策略配置
    eval_every = config['eval_every']  # 每500步评估一次
    validate_every = config['validate_every']  # 每500步验证一次
    save_every = config['save_every']  # 每500步保存一次

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)  # 创建策略
    # 是否加载预训练模型
    if config['load_pretrain']:
        loading_status = policy.deserialize(torch.load(
            os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'policy_step_50000_seed_0.ckpt')))
        print(f'loaded! {loading_status}')
    #  是否加载之前的模型
    if config['resume_ckpt_path'] is not None:
        loading_status = policy.deserialize(torch.load(config['resume_ckpt_path']))
        print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
    policy.cuda()  # 将模型放到GPU上
    optimizer = make_optimizer(policy_class, policy)  # 优化器

    min_val_loss = np.inf  # 最小验证损失 初始化为无穷大
    best_ckpt_info = None  # 最佳检查点信息 初始化为None

    train_dataloader = repeater(train_dataloader)  # 重复数据加载器/函数生成器
    # 开始训练----轮次
    for step in tqdm(range(num_steps + 1)):
        # validation 默认500
        if step % validate_every == 0:
            print('validating')

            with torch.inference_mode():
                policy.eval()  # 评估模式
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    # 前向传播
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break

                validation_summary = compute_dict_mean(validation_dicts)  # 计算均值
                epoch_val_loss = validation_summary['loss']  # 获取平均损失
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
            # 保存验证结果 并打印
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)
            wandb.log(validation_summary, step=step)
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

        # evaluation
        if (step > 0) and (step % eval_every == 0):
            # first save then eval
            ckpt_name = f'policy_step_{step}_seed_{seed}.ckpt'
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save(policy.serialize(), ckpt_path)
            success, _ = eval_bc(config, ckpt_name, arm_left, arm_right, serial_numbers, save_episode=True,
                                 num_rollouts=1, step=step)
            wandb.log({'success': success}, step=step)

        # training
        policy.train()  # 训练模式
        optimizer.zero_grad()  # 梯度清零
        data = next(train_dataloader)  # 获取下一个数据
        forward_dict = forward_pass(data, policy)  # 前向传播
        # backward
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        wandb.log(forward_dict, step=step)  # not great, make training 1-2% slower

        if step % save_every == 0:  # save_every = 500
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)

    # ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    ckpt_path = os.path.join(ckpt_dir, f'policy_step_500_seed_0.ckpt')
    torch.save(policy.serialize(), ckpt_path)

    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_step_{best_step}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}')

    return best_ckpt_info


# 重复数据加载器/函数生成器
def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, default='ACT', help='policy_class, capitalize',
                        required=False)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, default=16, help='batch_size', required=False)
    parser.add_argument('--seed', action='store', type=int, default=0, help='seed', required=False)
    parser.add_argument('--num_steps', action='store', type=int, default=10000, help='num_steps', required=False)
    parser.add_argument('--lr', action='store', type=float, default=1e-5, help='lr', required=False)
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--eval_every', action='store', type=int, default=500, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every',
                        required=False)
    parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, default=10, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, default=100, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, default=512, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, default=3200, help='dim_feedforward',
                        required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')

    arm_left = "REPLACE_WITH_LEFT_REALMAN_ARM_HOST"
    arm_right = "REPLACE_WITH_RIGHT_REALMAN_ARM_HOST"
    serial_numbers = [
        'REPLACE_WITH_HIGH_CAMERA_SERIAL',
        'REPLACE_WITH_LEFT_WRIST_CAMERA_SERIAL',
        'REPLACE_WITH_RIGHT_WRIST_CAMERA_SERIAL',
    ]

    main(vars(parser.parse_args()), arm_left, arm_right, serial_numbers)
