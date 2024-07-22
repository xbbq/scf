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

import numpy as np

# 原始矩阵 A
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# 目标矩阵 B，尺寸大于 A
B = np.zeros((A.shape[0], 10))  # 假设 B 是一个更大的矩阵，这里只是示例

# 索引矩阵 indices，表示从 A 复制元素到 B 的列位置
# 假设 indices 的每一行的元素都是要复制到 B 的列索引
indices = np.array([2, 1, 0])

# 使用高级索引填充 B
B[np.arange(A.shape[0]), indices] = A.T  # 注意：这里使用了 A.T 来转置 A

print("矩阵 A:")
print(A)
print("矩阵 B:")
print(B)



