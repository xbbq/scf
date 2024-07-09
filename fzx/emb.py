import torch
import torch.nn.functional as F
from torch import nn


class AutoDiscretizationEmbedding2(nn.Module):
    def __init__(self, dim, max_seq_len, bin_num, bin_alpha, mask_token_id = None, pad_token_id = None):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha     #系数
        
        #初始化两个全连接层mlp和mlp2，第一个层的输入维度为1，输出维度为分箱数量，第二个层的输入和输出维度都是分箱数量。
        #初始化LeakyReLU激活函数和Softmax归一化函数。
        #初始化一个嵌入层emb，它将分箱索引映射到嵌入维度。
        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)
        
        #初始化两个特殊嵌入层emb_mask和emb_pad，用于表示掩码和填充标记的嵌入。
        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)
        
        #创建一个包含分箱索引的Tensor。保存掩码和填充标记的ID。
        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        # print('self.bin_num_idx',self.bin_num_idx, self.bin_num_idx.shape)

        #创建一个值为0的Tensor，用于索引嵌入层的第一个元素。
        self.tensor0 = torch.tensor(0, dtype=torch.long)

    def forward(self, x, output_weight=0):

        #找出输入序列中掩码和填充标记的位置。
        x_mask_idx = (x==self.mask_token_id).nonzero()
        x_pad_idx = (x==self.pad_token_id).nonzero()
        # print("x_mask",x_mask_idx.shape,x_mask_idx)
        
        x = self.mlp(x) # [B,N,1] -> [B,N,H]
        x = self.LeakyReLU(x) # [B,N,H]
        x_crosslayer = self.mlp2(x) # [B,N,H]
        x = self.bin_alpha * x + x_crosslayer # [B,N,H]
        weight = self.Softmax(x) # [B, N, H]
        # print('weight', weight.shape, weight, torch.sum(weight, 2))
        
        #将分箱索引移动到与输入序列相同的设备上。
        bin_num_idx = self.bin_num_idx.to(x.device) # [H,]
        # print('bin_num_idx', bin_num_idx.shape)
        
        #获取分箱的嵌入表示。
        token_emb = self.emb(bin_num_idx) # [H, D]
        # print('token_emb', token_emb.shape)
        
        #通过矩阵乘法计算最终的嵌入序列。
        x = torch.matmul(weight, token_emb) #[B, N, D]
        # print("x_emb",x.shape,x)
        
        #在与输入序列相同的设备上创建一个值为0的Tensor。
        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)


        #获取掩码标记的嵌入表示，并将其替换到嵌入序列中的掩码位置。
        mask_token_emb = self.emb_mask(tensor0).to(x.device).type(x.dtype)
        # print(mask_token_emb.dtype)
        # print("x", x.dtype)


        #使用掩码标记的位置索引 x_mask_idx 来定位输出嵌入序列 x 中掩码标记应处的位置。
        #将掩码标记的嵌入表示 mask_token_emb 重复 x_mask_idx.shape[0] 次（即掩码标记的数量），以便它可以被放置在所有掩码标记的位置上。
        #将重复后的掩码标记嵌入表示放置在输出嵌入序列 x 的正确位置上。
        x[x_mask_idx[:,0],x_mask_idx[:,1],:] = mask_token_emb.repeat(x_mask_idx.shape[0],1)
        # print("x_emb",x.shape,x)

        #同上
        pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
        x[x_pad_idx[:,0],x_pad_idx[:,1],:] = pad_token_emb.repeat(x_pad_idx.shape[0],1)
    
        if output_weight:
            return x,weight
        return x
