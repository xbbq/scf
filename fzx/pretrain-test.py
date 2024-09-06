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
from scModeltest import scModel
from utils import detail,CosineAnnealingWarmupRestarts
from torch.optim import Adam
from torch import nn

#'/home/share/huadjyin/home/fengzhixin/scbert/data/panglao_human.h5ad'
BATCH_SIZE = 2
SEED = 0
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
        return full_seq

    def __len__(self):
        return self.data.shape[0]

data = sc.read_h5ad('C:\\Users\\fengzhixin\\Documents\\scfoundation\\scfoundation\\fzx\\data\\panglao_10000.h5ad')
# data = sc.read_h5ad('/home/share/huadjyin/home/fengzhixin/scf/scf/fzx/data/panglao_10000.h5ad')
data = data.X
# print(data.shape)
# 将其转换为稠密矩阵
data = data.toarray()
print('44','data.shape',data.shape)
# t = torch.cat((torch.from_numpy(data),torch.from_numpy(masked_matrix)),dim=1)


# print('data',data.shape)  #(10000, 3516)
# print('mask_positions',mask_positions.shape)
# print('not_zero_position',not_zero_position.dtype)
# print('masked_matrix',masked_matrix.shape)
# print('encoder_position_gene_ids',encoder_position_gene_ids.shape)
# # print('type(data)',type(data))

data_train, data_val = train_test_split(data, test_size=0.05,random_state=SEED)

train_dataset = SCDataset(data_train)

# print('-----------------------',index)
val_dataset = SCDataset(data_val)
train_sampler = RandomSampler(train_dataset)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,sampler=train_sampler)
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型实例
model = scModel(device, input_dim, data.shape[1], encoder_dim, decoder_dim, output_dim, num_encoder_layers, num_decoder_layers, 
                num_encoder_heads, num_decoder_heads, dropout,'mask_positions','not_zero_position','encoder_position_gene_ids',bin_num,bin_alpha)




# 将模型移至GPU
model.to(device)

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


for i in range(1, 100):
    # train_loader.sampler.set_epoch(i)
    print(f'    ==  Epoch: {i} ' )
    model.train()
    running_loss = 0.0
    cum_acc = 0.0
    for index, data in enumerate(train_loader):
        index += 1
        # print('146','dataindex',dataindex)
        print('147','data.shape',data.shape)
        ori_data = data
        data = data.numpy()
        unmasked_only_matrix,mask_positions,masked_matrix,not_zero_position, encoder_position_gene_ids = get_unmasked_only_matrix(data)
        print('unmasked_only_matrix.shape',unmasked_only_matrix.shape)
        print('mask_positions.shape',mask_positions.shape)
        print('masked_matrix.shape',masked_matrix.shape)
        print('not_zero_position.shape',not_zero_position.shape)
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
            # with model.no_sync():
            logits = model(data,full,mask_positions,encoder_position_gene_ids)
            logits = torch.squeeze(logits)
            print('shape1',logits.shape)
            print('shape2',ori_data.shape)
            loss = mse_loss(logits,ori_data) / GRADIENT_ACCUMULATION

            # gpu_tensor = data[0]
            # cpu_index = gpu_tensor.cpu()
            # tensor = torch.from_numpy(mask_positions[cpu_index])
            tensor = mask_positions.to(device)
            loss.to(device)
            # print('tensor',tensor)
            # print('loss',loss)

            loss = loss * tensor

            # tensor2 = torch.from_numpy(mask_positions[cpu_index])
            tensor2 = mask_positions.to(device)
            loss[tensor2 == 0] = 0
            loss = torch.mean(loss)

            loss.backward()
        if index % GRADIENT_ACCUMULATION == 0:
            logits = model(data,full,mask_positions,encoder_position_gene_ids)
            logits = torch.squeeze(logits)
            print('shape1',logits.shape)
            print('shape2',ori_data.shape)
            loss = mse_loss(logits,ori_data) / GRADIENT_ACCUMULATION

            tensor = mask_positions.to(device)
            loss.to(device)
            # print('tensor',tensor)
            # print('loss',loss)

            loss = loss * tensor

            # tensor2 = torch.from_numpy(mask_positions[cpu_index])
            tensor2 = mask_positions.to(device)
            loss[tensor2 == 0] = 0
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
    # epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
    # epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
    # if is_master:
    #     print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
    # dist.barrier()
    print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f}   ==')
    # 打开文件，模式为 'w'（写模式）
    with open("/home/share/huadjyin/home/fengzhixin/scf/scf/q.txt", "a") as file:
        file.write(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f}  ==')
        file.write("\n")  # 追加一个换行符
    print(1)
    scheduler.step()
    print(2)

    # if i % 1 == 0:
    #     model.eval()
    #     running_loss = 0.0
    #     running_error = 0.0
    #     predictions = []
    #     truths = []
    #     with torch.no_grad():
    #         for index, data in enumerate(val_loader):
    #             index += 1
    #             data = data.to(device)
    #             data, labels = data_mask(data)
    #             logits = model(data)
    #             loss = loss_fn(logits.transpose(1, 2), labels)
    #             running_loss += loss.item()
    #             softmax = nn.Softmax(dim=-1)
    #             final = softmax(logits)[..., 1:-1]
    #             final = final.argmax(dim=-1) + 1
    #             predictions.append(final)
    #             truths.append(labels)
    #         del data, labels, logits, final
    #         # gather
    #         predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
    #         truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
    #         correct_num = ((truths != PAD_TOKEN_ID) * (predictions == truths)).sum(dim=-1)[0].item()
    #         val_num = (truths != PAD_TOKEN_ID).sum(dim=-1)[0].item()
    #         val_loss = running_loss / index
    #         val_loss = get_reduced(val_loss, local_rank, 0, world_size)
    #     if is_master:
    #         val_acc = 100 * correct_num / val_num
    #         print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | Accuracy: {val_acc:6.4f}%  ==')
    # del predictions, truths

    # if is_master:
    #     save_ckpt(i, model, optimizer, scheduler, epoch_loss, model_name, ckpt_dir)