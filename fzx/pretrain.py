from torch.utils.data import DataLoader, Dataset
import random
import torch
import scanpy as sc
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
from utils import *
from preprocess import get_unmasked_only_matrix
import numpy as np
import os
import scipy.io  
import h5py


BATCH_SIZE = 2
SEED = 0

class SCDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start]
        return full_seq

    def __len__(self):
        return self.data.shape[0]
    # file_path = 'path_to_your_file.txt'

if os.path.exists('C:\\Users\\fengzhixin\\Documents\\scfoundation\\scfoundation\\output.h5ad'):
    # 指定文件名
    file_name = 'C:\\Users\\fengzhixin\\Documents\\scfoundation\\scfoundation\\output.h5ad'

    # 打开HDF5文件
    with h5py.File(file_name, 'r') as f:
        # 访问名为'unmasked_only_matrix'的数据集
        dataset = f['unmasked_only_matrix']['data']
        
        # 读取数据集的内容
        data = dataset[()]
    # data = sc.read_h5ad('C:\\Users\\fengzhixin\\Documents\\scfoundation\\scfoundation\\output.h5ad')
    # data = data.X
else:
    data = sc.read_h5ad('C:\\Users\\fengzhixin\\Documents\\scfoundation\\scfoundation\\fzx\\data\\panglao_10000.h5ad')
    data = data.X
    # print(data.shape)
    # 将其转换为稠密矩阵
    data = data.toarray()
    print(data.shape)
    unmasked_only_matrix = get_unmasked_only_matrix(data)
    print('unmasked_only_matrix.shape',unmasked_only_matrix.shape)
    print('got unmasked_only_matrix')

    # 指定文件名
    file_name = 'output.h5ad'

    # 使用scipy.io.savemat保存矩阵
    # scipy.io.savemat(file_name, {'unmasked_only_matrix': unmasked_only_matrix})

    # print(f'Matrix saved to {file_name}')
    # 创建HDF5文件
    with h5py.File(file_name, 'w') as f:
        # 创建一个名为'unmasked_only_matrix'的组
        group = f.create_group('unmasked_only_matrix')
        
        # 将矩阵数据写入组
        group.create_dataset('data', data=unmasked_only_matrix)
    
    # 指定文件名
    file_name = 'C:\\Users\\fengzhixin\\Documents\\scfoundation\\scfoundation\\output.h5ad'

    # 打开HDF5文件
    with h5py.File(file_name, 'r') as f:
        # 访问名为'unmasked_only_matrix'的数据集
        dataset = f['unmasked_only_matrix']['data']
        
        # 读取数据集的内容
        data = dataset[()]

print(data.shape)  #(10000, 3516)
print('type(data)',type(data))

data_train, data_val = train_test_split(data, test_size=0.05,random_state=SEED)

train_dataset = SCDataset(data_train)
val_dataset = SCDataset(data_val)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

data2 = sc.read_h5ad('C:\\Users\\fengzhixin\\Documents\\scfoundation\\scfoundation\\fzx\\data\\panglao_10000.h5ad')
data2 = data2.X
print('type(data2)',type(data2))

#todo get_emb
for index, data in enumerate(train_loader):
    # 检查data的类型
    if isinstance(data, torch.Tensor):
        # 如果是PyTorch张量，转换为NumPy数组
        data = data.numpy()


    print(data.shape)
    if index >10:
        break