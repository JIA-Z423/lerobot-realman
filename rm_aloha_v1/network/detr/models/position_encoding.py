# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor

import IPython

e = IPython.embed


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    num_pos_feats : 位置嵌入的维度（特征数量的一半，因为在代码中每个位置生成两个特征）
    temperature : 一个控制位置嵌入频率的参数，用来缩放嵌入值
    normalize ： 决定是否标准化嵌入，标准化可以使不同尺寸的输入特征在相同的范围内产生嵌入值
    scale ： 如果标准化，则使用该缩放因子。默认情况下，使用 2 * math.pi 作为缩放因子。
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    # 要对每个像素进行编码 像素对应两个方向的编码 一个是横向 一个是纵向 所以构造出来的1*H*W*N
    def forward(self, tensor):
        # tensor 大小：(batch_size, channels, height, width)
        x = tensor
        # mask = tensor_list.mask
        # assert mask is not None
        # not_mask = ~mask
        # 提取 第一个样本 的 第一个通道的数据  也就是生成not_mask维度为1*H*W的矩阵
        # PyTorch：这种写法是利用高级索引的功能来选择张量中的特定维度或元素。  x[0,0]表示选择第一个样本的第一个通道的数据 大小为H*W
        # 这种写法 第一个 会丢掉批次维度 但是第二个0 会保留通道维度
        not_mask = torch.ones_like(x[0, [0]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # 竖着累加
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # 横着累加
        # 归一化 把元素放缩到[0,scale]
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 生成位置编码 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, ...., num_pos_feats - 1]
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # 向下取整[0,0,1,1,2,2,3,3,.......]
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # 位置编码的维度为 (batch_size, num_pos_feats * 2, height, width)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


# 构建位置编码 选择正弦位置编码或者可学习位置编码
def build_position_encoding(args):
    # hidden_dim = 512 ====> N_steps = 256
    N_steps = args.hidden_dim // 2
    # 正弦位置编码 main.py:  parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    # 可学习位置编码
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
