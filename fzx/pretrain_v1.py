from torch.utils.data import DataLoader, Dataset,RandomSampler
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
from scModel_v1 import scModel
from utils import detail,CosineAnnealingWarmupRestarts,seed_all,get_reduced
from torch.optim import Adam
from torch import nn
import argparse
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import GradScaler

#'/home/share/huadjyin/home/fengzhixin/scbert/data/panglao_human.h5ad'
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help='Local process rank.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--batch_size", type=int, default=4, help='Number of batch size.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')

args = parser.parse_args()
# rank = int(os.environ["RANK"])
rank = int(os.environ.get('LOCAL_RANK',0))
local_rank = int(os.environ.get('LOCAL_RANK',0))
is_master = rank == 0
BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
SEED = args.seed
# local_rank = args.local_rank

dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = torch.distributed.get_world_size()
seed_all(SEED + torch.distributed.get_rank())
# datapath = 'C:\\Users\\fengzhixin\\Documents\\scfoundation\\scfoundation\\output.h5ad'
datapath = '/home/share/huadjyin/home/fengzhixin/scf/scf/output.h5ad'
GRADIENT_ACCUMULATION = 100

def cleanup():
    dist.destroy_process_group()

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

# data = sc.read_h5ad('C:\\Users\\fengzhixin\\Documents\\scfoundation\\scfoundation\\fzx\\data\\panglao_10000.h5ad')
data = sc.read_h5ad('/home/share/huadjyin/home/fengzhixin/scbert/data/panglao_human.h5ad')
# data = sc.read_h5ad('/home/share/huadjyin/home/fengzhixin/scf/scf/fzx/data/panglao_10000.h5ad')
tmp = data.X
half_index = tmp.shape[0] // 6
first_half = tmp[:half_index, :]
del data,tmp
# print(data.shape)
data = first_half.toarray()
del first_half
detail('data',data)
print('44','data.shape',data.shape)
data = data.astype(np.float16)
# t = torch.cat((torch.from_numpy(data),torch.from_numpy(masked_matrix)),dim=1)

data_train, data_val = train_test_split(data, test_size=0.05,random_state=SEED)

train_dataset = SCDataset(data_train)

# print('-----------------------',index)
val_dataset = SCDataset(data_val)
train_sampler = DistributedSampler(train_dataset)
val_sampler = SequentialDistributedSampler(val_dataset, batch_size=BATCH_SIZE, world_size=world_size)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,drop_last=True)

# data2 = sc.read_h5ad('C:\\Users\\fengzhixin\\Documents\\scfoundation\\scfoundation\\fzx\\data\\panglao_10000.h5ad')
# data2 = data2.X
# print('type(data2)',type(data2))
# 定义模型参数
input_dim = 16908  # 词汇表大小
encoder_dim = 768   # Encoder的嵌入维度
decoder_dim = 512   # Decoder的嵌入维度
output_dim = 1   # 输出维度，例如，下一个词的预测或分类任务的类别数
num_encoder_layers = 6  # Transformer Encoder层数
num_decoder_layers = 6    # Transformer Decoder层数
num_encoder_heads = 8             # 注意力机制中的头数
num_decoder_heads = 8             # 注意力机制中的头数
dropout = 0.1            # Dropout比率
bin_num=10, 
bin_alpha=1.0,
pad_token_id=input_dim+1, 
mask_token_id=input_dim+2
print('data.shape[1]',data.shape[1])
encoder_len = data.shape[1]

# 检查CUDA是否可用，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化GradScaler
scaler = GradScaler()

# 创建模型实例
model = scModel(device, input_dim, 1000, encoder_dim, decoder_dim, output_dim, num_encoder_layers, num_decoder_layers, 
                num_encoder_heads, num_decoder_heads, dropout,'mask_positions','not_zero_position','encoder_position_gene_ids',bin_num,bin_alpha)




# 将模型移至GPU
model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
# 将模型转换为半精度浮点数
# model.half()
# 确认模型已经在GPU上
print(f"Model is on {device}")


# optimizer
optimizer = Adam(model.parameters(), lr=1e-4)
# learning rate scheduler
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=1e-4,
    min_lr=1e-6,
    warmup_steps=5,
    gamma=0.9
)
loss_fn = nn.CrossEntropyLoss(reduction='mean')
# 计算 MSE 损失
mse_loss = nn.MSELoss(reduction='none')  # 使用 reduction='none' 来获取原始的损失矩阵
softmax = nn.Softmax(dim=-1)


# c = 0
# for index,data in enumerate(train_loader):
#     a = data
#     print('data.shape',data[1].shape)#[tensor([1032, 5836]), tensor([[ 1.6357,  1.8735,  2.0654,  ..., -2.0000, -2.0000, -2.0000],[ 1.5172,  1.0223,  1.0223,  ..., -2.0000, -2.0000, -2.0000]])]
#     y = model(data)
#     y = torch.squeeze(y)
#     print('y',y)
#     print('a',a)
#     detail('y',y)
#     c+=1
    
#     # get_emb(data)
#     if c >0:
#         break

dist.barrier()
for i in range(1, 100):
    train_loader.sampler.set_epoch(i)
    print(f'    ==  Epoch: {i} ' )
    model.train()
    dist.barrier()
    running_loss = 0.0
    cum_acc = 0.0
    for index, data in enumerate(train_loader):
        index += 1
        # print('146','dataindex',dataindex)
        print('147','data.shape',data.shape)
        data = data.numpy()
        ts_matrix,unmasked_only_matrix,mask_positions,masked_matrix,_, encoder_position_gene_ids = get_unmasked_only_matrix(data)
        detail('ts_matrix',ts_matrix)
        print('unmasked_only_matrix.shape',unmasked_only_matrix.shape)
        print('mask_positions.shape',mask_positions.shape)
        print('masked_matrix.shape',masked_matrix.shape)
        print('encoder_position_gene_ids.shape',encoder_position_gene_ids.shape)
        print('got unmasked_only_matrix')
        # data = data.to(device)
        data = torch.from_numpy(unmasked_only_matrix).to(device)
        mask_positions = torch.from_numpy(mask_positions).to(device)
        full = torch.from_numpy(masked_matrix).to(device)
        encoder_position_gene_ids = torch.from_numpy(encoder_position_gene_ids).to(device)

        # data = [tensor.to(device) for tensor in data]
        # data, labels = data_mask(data)
        if index % GRADIENT_ACCUMULATION != 0:
            with model.no_sync():
                with torch.cuda.amp.autocast():
                    logits = model(data,full,mask_positions,encoder_position_gene_ids)
                    logits = torch.squeeze(logits)
                    print('shape1',logits.shape)#16908
                    print('shape2',ts_matrix.shape)#16906
                    loss = mse_loss(logits,torch.from_numpy(ts_matrix).to(device)) / GRADIENT_ACCUMULATION
                    tensor = mask_positions.to(device)
                    loss.to(device)
                    loss = loss * tensor
                    tensor2 = mask_positions.to(device)
                    loss[tensor2 == 0] = 0
                    loss = torch.mean(loss)

                    loss.backward()
                # 使用GradScaler来缩放损失以避免梯度下溢
                # scaler.scale(loss).backward()
        if index % GRADIENT_ACCUMULATION == 0:
            with torch.cuda.amp.autocast():
                logits = model(data,full,mask_positions,encoder_position_gene_ids)
                logits = torch.squeeze(logits)
                print('shape1',logits.shape)
                print('shape2',torch.from_numpy(ts_matrix).shape)
                loss = mse_loss(logits,torch.from_numpy(ts_matrix).to(device)) / GRADIENT_ACCUMULATION
                tensor = mask_positions.to(device)
                loss.to(device)
                loss = loss * tensor
                tensor2 = mask_positions.to(device)
                loss[tensor2 == 0] = 0
                loss = torch.mean(loss)
                loss.backward()
            # 使用GradScaler来缩放损失以避免梯度下溢
            # scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e2))
                optimizer.step()
            # 更新模型权重
            # scaler.step(optimizer)
            # scaler.update()  # 更新scaler的缩放比例
                optimizer.zero_grad()
        running_loss += loss.item()
    epoch_loss = running_loss / index
    epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
    # cleanup()
    if is_master:
        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f}  ==')
        with open("/home/share/huadjyin/home/fengzhixin/scf/scf/m0926.txt", "a") as file:
            file.write(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.9f}  ==')
            file.write("\n")  # 追加一个换行符
    dist.barrier()
    scheduler.step()

    if i % 1 == 0:
        model.eval()
        dist.barrier()
        running_loss = 0.0

        with torch.no_grad():
            for index, data in enumerate(val_loader):
                index += 1
                data = data.numpy()
                ts_matrix,unmasked_only_matrix,mask_positions,masked_matrix,_, encoder_position_gene_ids = get_unmasked_only_matrix(data)
                data = torch.from_numpy(unmasked_only_matrix).to(device)
                mask_positions = torch.from_numpy(mask_positions).to(device)
                full = torch.from_numpy(masked_matrix).to(device)
                encoder_position_gene_ids = torch.from_numpy(encoder_position_gene_ids).to(device)

                logits = model(data,full,mask_positions,encoder_position_gene_ids)
                logits = torch.squeeze(logits)
                print('shape1',logits.shape)
                print('shape2',torch.from_numpy(ts_matrix).shape)
                loss = mse_loss(logits,torch.from_numpy(ts_matrix).to(device)) / GRADIENT_ACCUMULATION
                tensor = mask_positions.to(device)
                loss.to(device)
                loss = loss * tensor
                tensor2 = mask_positions.to(device)
                loss[tensor2 == 0] = 0
                loss = torch.mean(loss)
                running_loss += loss.item()
            # gather
            val_loss = running_loss / index
            val_loss = get_reduced(val_loss, local_rank, 0, world_size)
        if is_master:
            print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} ')
            with open("/home/share/huadjyin/home/fengzhixin/scf/scf/m0926.txt", "a") as file:
                file.write(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.9f} ')
                file.write("\n")  # 追加一个换行符

    # if is_master:
    #     save_ckpt(i, model, optimizer, scheduler, epoch_loss, model_name, ckpt_dir)