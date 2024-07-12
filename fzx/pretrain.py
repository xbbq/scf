from torch.utils.data import DataLoader, Dataset
import random
import torch
import scanpy as sc
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
from utils import *
from preprocess import test
import numpy as np


BATCH_SIZE = 2
SEED = 0

class SCDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        return full_seq

    def __len__(self):
        return self.data.shape[0]

data = sc.read_h5ad('C:\\Users\\fengzhixin\\Documents\\scfoundation\\scfoundation\\fzx\\data\\Zheng68K.h5ad')
data = data.X
print(data.shape)
# 将其转换为稠密矩阵
data = data.toarray()
print(data.shape)
# test(data)
data_train, data_val = train_test_split(data, test_size=0.05,random_state=SEED)

train_dataset = SCDataset(data_train)
val_dataset = SCDataset(data_val)

# train_sampler = DistributedSampler(train_dataset)
# val_sampler = SequentialDistributedSampler(val_dataset, batch_size=BATCH_SIZE, world_size=world_size)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# for index, data in enumerate(train_loader):
    # print(data.shape)
    # test(data)