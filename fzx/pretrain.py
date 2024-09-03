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
from scModel import scModel
from utils import detail,CosineAnnealingWarmupRestarts,seed_all,get_reduced
from torch.optim import Adam
from torch import nn
import argparse
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils import *

#'/home/share/huadjyin/home/fengzhixin/scbert/data/panglao_human.h5ad'
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help='Local process rank.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--batch_size", type=int, default=2, help='Number of batch size.')
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
GRADIENT_ACCUMULATION = 20

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
    #'/home/share/huadjyin/home/fengzhixin/scbert/data/panglao_human.h5ad'
    # data = sc.read_h5ad('C:\\Users\\fengzhixin\\Documents\\scfoundation\\scfoundation\\fzx\\data\\panglao_10000.h5ad')
    data = sc.read_h5ad('/home/share/huadjyin/home/fengzhixin/scf/scf/fzx/data/panglao_10000.h5ad')
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
val_dataset = SCDataset(data_val)
# train_sampler = RandomSampler(train_dataset)


# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,sampler=train_sampler)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
#--------------------------------------------------------------------------------------------------
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
num_encoder_layers = 12  # Transformer Encoder层数
num_decoder_layers = 6    # Transformer Decoder层数
num_encoder_heads = 12             # 注意力机制中的头数
num_decoder_heads = 8             # 注意力机制中的头数
dropout = 0.1            # Dropout比率
bin_num=10, 
bin_alpha=1.0,
pad_token_id=input_dim+1, 
mask_token_id=input_dim+2
print('data.shape[1]',data.shape[1])
encoder_len = data.shape[1]

# 检查CUDA是否可用，并设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型实例
model = scModel(device, input_dim, data.shape[1], encoder_dim, decoder_dim, output_dim, num_encoder_layers, num_decoder_layers, 
                num_encoder_heads, num_decoder_heads, dropout,mask_positions,not_zero_position,encoder_position_gene_ids,bin_num,bin_alpha)




# 将模型移至GPU
model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
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
# 计算 MSE 损失
mse_loss = nn.MSELoss(reduction='none').to(local_rank)  # 使用 reduction='none' 来获取原始的损失矩阵
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
for i in range(1, EPOCHS+1):
    train_loader.sampler.set_epoch(i)
    print(f'    ==  Epoch: {i} ' )
    model.train()
    dist.barrier()
    running_loss = 0.0
    cum_acc = 0.0
    for index, data in enumerate(train_loader):
        index += 1
        # data = data.to(device)
        data = [tensor.to(device) for tensor in data]
        # data, labels = data_mask(data)
        if index % GRADIENT_ACCUMULATION != 0:
            # with model.no_sync():
            logits = model(data)
            logits = torch.squeeze(logits)
            print('shape1',logits.shape)
            print('shape2',data[1][:,encoder_len:].shape)
            loss = mse_loss(logits,data[1][:,encoder_len:]) / GRADIENT_ACCUMULATION

            # detail('mask_positions[data[0]]',mask_positions[data[0]])
            
            # loss_detached = loss.detach().numpy()
            # loss_detached = loss_detached * mask_positions[data[0]]
            # detail('loss_detached',loss_detached)
            # loss_detached[mask_positions == 0] = 0
            # loss = torch.mean(loss_detached)

            # gpu_tensor = index
            # cpu_index = gpu_tensor.cpu()
            # # detail('cpu_encoder_position_gene_ids',cpu_encoder_position_gene_ids)
            # tensor = torch.from_numpy(self.encoder_position_gene_ids[cpu_index])
            # tensor = tensor.to(self.device)
            # position_emb = self.pos_emb(tensor)

            gpu_tensor = data[0]
            cpu_index = gpu_tensor.cpu()
            tensor = torch.from_numpy(mask_positions[cpu_index])
            tensor = tensor.to(device)
            loss.to(device)
            print('tensor',tensor)
            print('loss',loss)

            loss = loss * tensor

            tensor2 = torch.from_numpy(mask_positions[cpu_index])
            tensor2 = tensor2.to(device)
            loss[tensor2 == 0] = 0
            loss = torch.mean(loss)

            loss.backward()
        if index % GRADIENT_ACCUMULATION == 0:
            logits = model(data)
            logits = torch.squeeze(logits)
            loss = mse_loss(logits,data[1][:,encoder_len:]) / GRADIENT_ACCUMULATION

            gpu_tensor = data[0]
            cpu_index = gpu_tensor.cpu()
            tensor = torch.from_numpy(mask_positions[cpu_index])
            tensor = tensor.to(device)
            loss.to(device)
            loss = loss * tensor
            # loss = loss * torch.from_numpy(mask_positions[data[0]])
            tensor2 = torch.from_numpy(mask_positions[cpu_index])
            tensor2 = tensor2.to(device)
            loss[tensor2 == 0] = 0
            # loss[mask_positions[data[0]] == 0] = 0
            loss = torch.mean(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e2))
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item()
    #     final = softmax(logits)[..., 1:-1]
    #     final = final.argmax(dim=-1) + 1
    #     pred_num = (labels != PAD_TOKEN_ID).sum(dim=-1)
    #     correct_num = ((labels != PAD_TOKEN_ID) * (final == labels)).sum(dim=-1)
    #     cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
    epoch_loss = running_loss / index
    # epoch_acc = 100 * cum_acc / index
    epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
    # epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
    if is_master:
        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f}  ==')
    dist.barrier()
    # print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f}   ==')
    # 打开文件，模式为 'w'（写模式）
    with open("/home/share/huadjyin/home/fengzhixin/scf/scf/p.txt", "a") as file:
        file.write(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f}  ==')
        file.write("\n")  # 追加一个换行符
    print(1)
    scheduler.step()
    print(2)

    if i % 1 == 0:
        model.eval()
        dist.barrier()
        running_loss = 0.0
        # running_error = 0.0
        # predictions = []
        # truths = []
        with torch.no_grad():
            for index, data in enumerate(val_loader):
                index += 1
                # data = data.to(device)
                data = [tensor.to(device) for tensor in data]
                # data, labels = data_mask(data)
                logits = model(data)
                logits = torch.squeeze(logits)
                loss = mse_loss(logits,data[1][:,encoder_len:])
                # loss = loss_fn(logits.transpose(1, 2), labels)
                gpu_tensor = data[0]
                cpu_index = gpu_tensor.cpu()
                tensor = torch.from_numpy(mask_positions[cpu_index])
                tensor = tensor.to(device)
                loss.to(device)
                loss = loss * tensor
                # loss = loss * torch.from_numpy(mask_positions[data[0]])
                tensor2 = torch.from_numpy(mask_positions[cpu_index])
                tensor2 = tensor2.to(device)
                loss[tensor2 == 0] = 0
                # loss[mask_positions[data[0]] == 0] = 0
                loss = torch.mean(loss)
                running_loss += loss.item()

            #     softmax = nn.Softmax(dim=-1)
            #     final = softmax(logits)[..., 1:-1]
            #     final = final.argmax(dim=-1) + 1
            #     predictions.append(final)
            #     truths.append(labels)
            # del data, labels, logits, final
            # gather
            # predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
            # truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
            # correct_num = ((truths != PAD_TOKEN_ID) * (predictions == truths)).sum(dim=-1)[0].item()
            # val_num = (truths != PAD_TOKEN_ID).sum(dim=-1)[0].item()
            val_loss = running_loss / index
            val_loss = get_reduced(val_loss, local_rank, 0, world_size)
        # print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f}   ==')
        # 打开文件，模式为 'w'（写模式）
        with open("/home/share/huadjyin/home/fengzhixin/scf/scf/p.txt", "a") as file:
            file.write(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f}   ==')
            file.write("\n")  # 追加一个换行符
        if is_master:
            print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f}   ==')
    # del predictions, truths

    # if is_master:
    #     save_ckpt(i, model, optimizer, scheduler, epoch_loss, model_name, ckpt_dir)