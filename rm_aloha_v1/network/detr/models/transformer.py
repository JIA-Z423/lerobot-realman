# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import IPython

e = IPython.embed


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        # 一个Transformer模型由一个编码器和一个解码器组成
        # 编码器由num_encoder_layers个编码器层组成
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # 解码器由num_decoder_layers个解码器层组成
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # tranformer的前向传播
    def forward(self, src, mask, query_embed, pos_embed, latent_input=None, proprio_input=None,
                additional_pos_embed=None):
        # TODO flatten only when input has H and W
        if len(src.shape) == 4:  # has H and W
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)  # 通过flatten将src从NxCxHxW变为HWxNxC
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)  # 位置编码
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # 查询编码 1, bs, dim
            # mask = mask.flatten(1)

            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1)  # seq, bs, dim
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)  # 拼接位置编码

            addition_input = torch.stack([latent_input, proprio_input], axis=0)  # 堆叠
            src = torch.cat([addition_input, src], axis=0)  # 拼接输入
        else:
            assert len(src.shape) == 3
            # flatten NxHWxC to HWxNxC
            bs, hw, c = src.shape
            src = src.permute(1, 0, 2)  # 通过permute将src从NxHWxC变为HWxNxC
            pos_embed = pos_embed.unsqueeze(1).repeat(1, bs, 1)  # 位置编码
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # 查询编码

        # 查询嵌入 at 目的：？
        tgt = torch.zeros_like(query_embed)  # 生成一个和query_embed相同大小的全零张量
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # 通过编码器
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)  # 通过解码器
        hs = hs.transpose(1, 2)  # 转置
        return hs


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # 复制num_layers（4）个encoder_layer
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # d_model = 512 nhead = 8 dropout = 0.1
        # at 实现多头注意力机制的类  nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0,....)
        # 输入向量的维度 多头注意力机制中的注意力头的数量 注意力权重上dropout的概率
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # 两个前馈神经网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # 归一化层 + 额外的dropout层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # 设置激活函数
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before  # 是否子层操作前进行归一化

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # 如果位置编码为空 则返回tensor 否则返回tensor + pos
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # 添加位置编码 ---->位置嵌入
        q = k = self.with_pos_embed(src, pos)
        # 计算多头注意力机制 并且只取第一个  Q self_attn会返包含两个元素的元组 第一个是结果 第二个是权重
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # dropout1
        src = self.norm1(src)  # 归一化
        # 前馈神经网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # dropout2 + 归一化
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    # at 上下两个函数的区别：归一化所在的位置
    #  at forward_post称为后归一化  Post-Normalization；
    #  at forward_pre 则是预归一化  Pre-Normalization
    # 预归一化: 在多头自注意力和前馈神经网络之前进行层归一化，使得每层的输入保持稳定，有助于训练过程的稳定性。
    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Q 为什么这里有两个多头注意力机制 ？？？
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # 两个前馈神经网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 三个归一化层 + 三个dropout层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # 添加位置编码 ---->位置嵌入
        q = k = self.with_pos_embed(tgt, query_pos)
        # 自注意力机制  掩码多头自注意力机制
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)  # dropout1  at 残差连接
        tgt = self.norm1(tgt)  # 归一化
        # 多头注意力机制
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)  # dropout2
        tgt = self.norm2(tgt)  # 归一化
        # 前馈神经网络
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # dropout3 + 归一化
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    '''
    用于构建Transformer模型
    参数:
    - args.hidden_dim: 模型的隐藏层维度，即d_model，表示Transformer中所有层的输入和输出特征的维度。
    - args.dropout: 在模型中使用的dropout比率，用于防止过拟合。
    - args.nheads: 多头注意力机制中的头数，即nhead，允许模型在不同的表示子空间中并行学习信息。
    - args.dim_feedforward: 前馈网络的维度，即dim_feedforward，表示隐藏层的大小。
    - args.enc_layers: 编码器中的层次数，即num_encoder_layers，定义了编码器内部的子层数量。
    - args.dec_layers: 解码器中的层次数，即num_decoder_layers，定义了解码器内部的子层数量。
    - args.pre_norm: 是否在注意力和前馈网络操作之前应用层归一化，即normalize_before，这个选项可以影响训练的稳定性和性能。
    - return_intermediate_dec: 是否在解码器中返回每个解码层的中间结果，对于某些应用（如可视化或特征提取）可能很有用。
    '''
    return Transformer(
        d_model=args.hidden_dim,  # 模型的隐藏层维度 512
        dropout=args.dropout,  # dropout比率 0.1
        nhead=args.nheads,  # 多头注意力机制中的头数 8
        dim_feedforward=args.dim_feedforward,  # 前馈网络的维度 2048
        num_encoder_layers=args.enc_layers,  # 编码器中的层次数 4
        num_decoder_layers=args.dec_layers,  # 解码器中的层次数 6
        normalize_before=args.pre_norm,  # 是否在注意力和前馈网络操作之前应用层归一化
        return_intermediate_dec=True,  # 是否在解码器中返回每个解码层的中间结果
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
