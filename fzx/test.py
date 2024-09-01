import random
import numpy as np
import scanpy as sc
from torch.utils.data import DataLoader, Dataset
import random
import torch
import scanpy as sc
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
from utils import *
from preprocess import get_unmasked_only_matrix,get_emb,matrix
import numpy as np
import os
import scipy.io  
import h5py

class SCDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start]
        return rand_start, full_seq

    def __len__(self):
        return self.data.shape[0]
    
# unmasked_only_matrix,mask_positions,not_zero_position = get_unmasked_only_matrix(matrix)
# print('unmasked_only_matrix.shape',unmasked_only_matrix.shape)
# print('mask_positions',mask_positions)
# print('not_zero_position',not_zero_position)
# print('got unmasked_only_matrix')

# train_dataset = SCDataset(unmasked_only_matrix)

# train_loader = DataLoader(train_dataset, batch_size=2)

# for index,data in enumerate(train_loader):
#     print('data.shape',data[0])#[tensor([1032, 5836]), tensor([[ 1.6357,  1.8735,  2.0654,  ..., -2.0000, -2.0000, -2.0000],[ 1.5172,  1.0223,  1.0223,  ..., -2.0000, -2.0000, -2.0000]])]
#     print(mask_positions[data[0][0]])
#     print(not_zero_position[data[0][0]])

import torch
import torch.nn as nn
import torch.nn.functional as F

class PerformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, kernel_size=512):
        super(PerformerDecoderLayer, self).__init__()
        # 标准的 Transformer 解码器层组件
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Performer 特有的组件
        self.random_features = nn.Linear(d_model, kernel_size)
        self.projection = nn.Linear(kernel_size, d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 标准的 Transformer 解码器自注意力
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        # Performer 风格的编码器-解码器注意力
        # 这里使用随机特征和正交化技巧来近似注意力机制
        # 为了简化，我们不展示这些技术的具体实现
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        # 标准的 Transformer 解码器前馈网络
        tgt2 = self.fc1(tgt)
        tgt2 = F.relu(tgt2)
        tgt2 = self.fc2(tgt2)
        tgt = tgt + self.dropout(tgt2)

        # 返回解码器层的输出
        return tgt

class PerformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super(PerformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            PerformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, memory_mask=None, tgt_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        tgt = self.norm(tgt)
        return tgt

# 使用示例
# d_model = 512
# num_layers = 6
# num_heads = 8
# d_ff = 2048
# dropout = 0.1

# decoder = PerformerDecoder(num_layers, d_model, num_heads, d_ff, dropout)
# tgt = torch.rand(10, 32, d_model)  # (seq_length, batch_size, d_model)
# memory = torch.rand(20, 32, d_model)  # (seq_length, batch_size, d_model)

# output = decoder(tgt, memory)
# print(output)

input = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                             [5.0, 4.0, 3.0, 2.0, 1.0]])
target = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                             [4.0, 4.0, 3.0, 2.0, 2.0]])

mask = torch.tensor([[1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 1]])

# 计算 MSE 损失
mse_loss = nn.MSELoss(reduction='none')  # 使用 reduction='none' 来获取原始的损失矩阵
loss = mse_loss(input, target)
print(loss)

# 应用掩码：将掩码中为 0 的位置设置为无穷大
loss = loss * mask
print(loss)
loss[mask == 0] = 0  # 将掩码为 0 的损失设置为无穷大
print(loss)

# 计算加权 MSE 损失，忽略掩码为 0 的位置
weighted_loss = torch.mean(loss)  # 求和后除以非无穷大的元素数量
print(weighted_loss)
# print(weighted_loss.item)
#'/home/share/huadjyin/home/fengzhixin/scbert/data/panglao_human.h5ad'
# data = sc.read_h5ad('data/panglao_10000.h5ad')

for i in range(loss.shape[0]):
    for j in range(loss.shape[1]):
        print(loss[i][j])



