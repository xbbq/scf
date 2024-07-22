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
from scModel import scModel
import utils
from utils import detail


BATCH_SIZE = 2
SEED = 0
datapath = 'C:\\Users\\fengzhixin\\Documents\\scfoundation\\scfoundation\\output.h5ad'

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

if os.path.exists(datapath):
    # # 指定文件名
    # file_name = datapath

    # 打开HDF5文件
    with h5py.File(datapath, 'r') as f:
        # 访问名为'unmasked_only_matrix'的数据集
        dataset = f['unmasked_only_matrix']['data']
        dataset2 = f['mask_positions']['data']
        dataset3 = f['not_zero_position']['data']
        dataset4 = f['masked_matrix']['data']
        dataset5 = f['encoder_position_gene_ids']['data']
        
        # 读取数据集的内容
        data = dataset[()]
        mask_positions = dataset2[()]
        not_zero_position = dataset3[()]
        masked_matrix = dataset4[()]
        encoder_position_gene_ids = dataset5[()]
        encoder_position_gene_ids = encoder_position_gene_ids.astype(int)
else:
    data = sc.read_h5ad('C:\\Users\\fengzhixin\\Documents\\scfoundation\\scfoundation\\fzx\\data\\panglao_10000.h5ad')
    data = data.X
    # print(data.shape)
    # 将其转换为稠密矩阵
    data = data.toarray()
    print(data.shape)
    unmasked_only_matrix,mask_positions,masked_matrix,not_zero_position, encoder_position_gene_ids = get_unmasked_only_matrix(data)
    print('unmasked_only_matrix.shape',unmasked_only_matrix.shape)
    print('mask_positions.shape',mask_positions.shape)
    print('masked_matrix.shape',masked_matrix.shape)
    print('not_zero_position.shape',not_zero_position.shape)
    print('encoder_position_gene_ids.shape',encoder_position_gene_ids.shape)
    print('got unmasked_only_matrix')

    # # 指定文件名
    # file_name = 'output.h5ad'
    # 创建HDF5文件
    with h5py.File(datapath, 'w') as f:
        # 创建一个名为'unmasked_only_matrix'的组
        group = f.create_group('unmasked_only_matrix')
        group2 = f.create_group('mask_positions')
        group3 = f.create_group('not_zero_position')
        group4 = f.create_group('masked_matrix')
        group5 = f.create_group('encoder_position_gene_ids')
        
        # 将矩阵数据写入组
        group.create_dataset('data', data=unmasked_only_matrix)
        group2.create_dataset('data', data=mask_positions)
        group3.create_dataset('data', data=not_zero_position)
        group4.create_dataset('data', data=masked_matrix)
        group5.create_dataset('data', data=encoder_position_gene_ids)
    
    # 指定文件名
    file_name = datapath

    # 打开HDF5文件
    with h5py.File(file_name, 'r') as f:
        # 访问名为'unmasked_only_matrix'的数据集
        dataset = f['unmasked_only_matrix']['data']
        dataset2 = f['mask_positions']['data']
        dataset3 = f['not_zero_position']['data']
        dataset4 = f['masked_matrix']['data']
        dataset5 = f['encoder_position_gene_ids']['data']
        
        # 读取数据集的内容
        data = dataset[()]
        mask_positions = dataset2[()]
        not_zero_position = dataset3[()]
        masked_matrix = dataset4[()]
        encoder_position_gene_ids = dataset5[()]
        encoder_position_gene_ids = encoder_position_gene_ids.astype(int)
print('type(data)',type(data))
print('type(mask_positions)',type(mask_positions))
print('type(not_zero_position)',type(not_zero_position))
print('type(masked_matrix)',type(masked_matrix))

t = torch.cat((torch.from_numpy(data),torch.from_numpy(masked_matrix)),dim=1)
print('t.shape',t.shape)
print('type(t)',type(t))

print('data',data.shape)  #(10000, 3516)
print('mask_positions',mask_positions.shape)
print('not_zero_position',not_zero_position.dtype)
print('masked_matrix',masked_matrix.shape)
print('encoder_position_gene_ids',encoder_position_gene_ids.shape)
# print('type(data)',type(data))

data_train, data_val = train_test_split(t, test_size=0.05,random_state=SEED)

train_dataset = SCDataset(data_train)

# print('-----------------------',index)
val_dataset = SCDataset(data_val)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# data2 = sc.read_h5ad('C:\\Users\\fengzhixin\\Documents\\scfoundation\\scfoundation\\fzx\\data\\panglao_10000.h5ad')
# data2 = data2.X
# print('type(data2)',type(data2))
# 定义模型参数
input_dim = 16908  # 词汇表大小
encoder_dim = 768   # Encoder的嵌入维度
decoder_dim = 512   # Decoder的嵌入维度
output_dim = 1   # 输出维度，例如，下一个词的预测或分类任务的类别数
num_encoder_layers = 12  # Transformer Encoder层数
num_decoder_layers = 6    # Transformer Decoder层数
num_heads = 8             # 注意力机制中的头数
dropout = 0.1            # Dropout比率
bin_num=10, 
bin_alpha=1.0,
pad_token_id=input_dim+1, 
mask_token_id=input_dim+2
print('data.shape[1]',data.shape[1])

# 创建模型实例
model = scModel(input_dim, data.shape[1], encoder_dim, decoder_dim, output_dim, num_encoder_layers, num_decoder_layers, 
                num_heads, dropout,mask_positions,not_zero_position,encoder_position_gene_ids,bin_num,bin_alpha)
#get_emb
c = 0
for index,data in enumerate(train_loader):
    print('data.shape',data[1].shape)#[tensor([1032, 5836]), tensor([[ 1.6357,  1.8735,  2.0654,  ..., -2.0000, -2.0000, -2.0000],[ 1.5172,  1.0223,  1.0223,  ..., -2.0000, -2.0000, -2.0000]])]
    y = model(data)
    y = torch.squeeze(y)
    print('y',y)
    detail('y',y)
    c+=1
    
    # get_emb(data)
    if c >0:
        break