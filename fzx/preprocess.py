import numpy as np
import pandas as pd
import scanpy as sc
import torch
from tqdm import tqdm
from scModel import AutoDiscretizationEmbedding2
from torch import nn

T_threshold = 1000


def Hierarchical_Bayesian_downsampling(matrix):
    T_list = []
    S_list = []

    C,N = matrix.shape
    # 生成C个bernoulli(0.5)分布的随机样本
    bernoulli_samples = np.random.binomial(1, 0.5, size=C)
    # 生成C个 β（2,2）分布的随机数
    b_list = np.random.beta(2, 2, C).tolist()

    for i in range(0, C):
        T_list.append(matrix[i].sum())
        # if bernoulli_samples[i] == 1:   #T<1000 舍去
        if matrix[i].sum() >= T_threshold and bernoulli_samples[i] == 1:   #T<1000 舍去
            for j in range(0, N):
                matrix[i][j] = np.random.binomial(matrix[i][j], b_list[i])
        S_list.append(matrix[i].sum())
    #计算S
    return matrix,T_list,S_list

# matrix = np.random.uniform(0, 400, size=(5, 6))

matrix = [[  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 221,   0,   0,   0,  0,   0],
[  0,   0,  51,   0,   0,   0,   0,   0,  60,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0],
[  0,   0,   0,   0,  82,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 332,   0,   0, 12,   0],
[349, 321,   0,   0,   0,   0,   0, 231,   0,   0,   0,   0,   0, 338,   0,   0,   0,   0, 0,   0]]
matrix = np.array(matrix)
#----------------------- 生成测试矩阵----------------
# matrix = np.zeros((20, 20), dtype=int)

# # np.random.seed(0)  # 设置随机种子以获得可重复的结果
# for _ in range(40):
#     i, j = np.random.randint(0, 20, size=2)
#     matrix[i, j] = np.random.randint(10, 400)  # 随机选择一个1到9之间的整数
# print(matrix)
# #-------------------------------------------------
# t_marix,tlist,slist = Hierarchical_Bayesian_downsampling(matrix)
# print(t_marix)
# print(tlist)
# print(slist)

#------------------------------------------------------
#在经过下采样后，19,264 个基因会经过Library-size Normalization (除以细胞总 counts 数) 和 Log-Transformation。
def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to select target genes encode protein 
    Parameters:
        adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
        gene_list->list: wanted target gene 
    Returns:
        adata_new->`~anndata.AnnData` object
        to_fill_columns->list: zero padding gene
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
                              columns=to_fill_columns, 
                              index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns,var
# gene_list_df = pd.read_csv('../model/OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
# gene_list = list(gene_list_df['gene_name'])
# if t_marix.shape[1]<19264:
#         print('covert gene feature into 19264')
#         t_marix, to_fill_columns,var = main_gene_selection(t_marix,gene_list)
#         assert gexpr_feature.shape[1]>=19264

# adata = sc.AnnData(t_marix)
# sc.pp.normalize_total(adata)
# sc.pp.log1p(adata)
# gexpr_feature = pd.DataFrame(adata.X,index=adata.obs_names,columns=adata.var_names)
# print('gexpr_feature\n',gexpr_feature)
# tmp = []
# for i in tqdm(range(gexpr_feature.shape[0])):
#     # 用totalcount = gexpr_feature.iloc[i,:].sum()计算T，无法计算S
#     # 直接对T、S进行log变换，不确定是否能准确表示表达量
#     # 暂时直接用log(1+x)
#     totalcount = np.log1p(tlist[i])
#     sourcecount = np.log1p(slist[i])
#     tmpdata = (gexpr_feature.iloc[i,:]).tolist()
#     pretrain_gene_x = torch.tensor(tmpdata+[totalcount,sourcecount]).unsqueeze(0)
#     # data_gene_ids = torch.arange(gexpr_feature.shape[1]+2, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)
#     tmp.append(pretrain_gene_x)
# print('tmp\n',tmp)


def random_mask_with_position(matrix, mask_ratio, mask_value=-1, pad_value=-2):
    """
    对矩阵的每一行元素进行随机遮盖，并返回两个矩阵：一个是原始值和遮盖标记的矩阵，另一个是被遮盖后的矩阵。
    非零值遮盖概率为零值遮盖概率的十倍。

    参数:
    matrix: 输入的矩阵
    mask_ratio: 非零遮盖比例，应该在0到1之间，零值遮盖比例为mask_ratio/10
    mask_value: 用于遮盖的值

    返回:
    包含原始值和遮盖标记的矩阵
    被遮盖后的矩阵
    """
    # 确保mask_ratio在合理的范围内
    if mask_ratio < 0 or mask_ratio > 1:
        raise ValueError("mask_ratio should be between 0 and 1")
    
    condition2 = (matrix != 0) 

    # 复制矩阵以避免修改原始数据
    masked_matrix = matrix.copy()
    mask_positions = np.zeros_like(matrix)

    # 遍历每一行
    for i in range(matrix.shape[0]):
        # 分离非零元素和零元素
        non_zero_indices = np.where(matrix[i, :-2] != 0)[0]
        zero_indices = np.where(matrix[i, :-2] == 0)[0]

        # 计算非零元素和零元素的遮盖数量
        non_zero_mask_num = int(np.ceil(non_zero_indices.size * mask_ratio))
        zero_mask_num = int(np.ceil(zero_indices.size * mask_ratio / 10))

        # 确保遮盖数量不会超过实际存在的元素数量
        non_zero_mask_num = min(non_zero_mask_num, non_zero_indices.size)
        zero_mask_num = min(zero_mask_num, zero_indices.size)

        # 随机选择非零元素进行遮盖
        if non_zero_mask_num > 0:
            non_zero_mask_indices = np.random.choice(
                non_zero_indices,
                non_zero_mask_num,
                replace=False
            )
            for index in non_zero_mask_indices:
                masked_matrix[i, index] = mask_value
                mask_positions[i, index] = 1  # 记录遮盖位置

        # 随机选择零元素进行遮盖
        if zero_mask_num > 0:
            zero_mask_indices = np.random.choice(
                zero_indices,
                zero_mask_num,
                replace=False
            )
            for index in zero_mask_indices:
                masked_matrix[i, index] = mask_value
                mask_positions[i, index] = 1  # 记录遮盖位置

    #-----------生成encoder需要的矩阵--------------------

    # 使用 where 函数生成一个布尔数组，其中 True 对应于非零和非-1的元素
    condition = (masked_matrix != 0) & (masked_matrix != mask_value)

    # # 确定列数，即满足条件的最大列数
    max_cols = np.max(condition, axis=1)

    # # 将 max_cols 转换为整数
    max_cols = np.array(max_cols, dtype=int)

    # 创建一个新的列表，用于存储每一行的非零向量
    non_zero_vectors = []

    # 遍历矩阵的每一行
    for row in masked_matrix:
        # 创建一个空向量，用于存储非零值
        non_zero_vector = []
        # 遍历行的每个元素
        for element in row:
            # 如果元素不是0，则添加到向量中
            if element != 0 and element != mask_value:
                non_zero_vector.append(element)
        #用pad标记-2占位
        if len(non_zero_vector) == 0:
            non_zero_vector.append(pad_value)
        # 将非零向量添加到列表中
        non_zero_vectors.append(non_zero_vector)

    # print("Original Matrix:")
    # print(masked_matrix)
    # print("New Vectors with Non-Zero Values:")
    # print(non_zero_vectors)
    # 找到向量的最大长度
    max_length = max(len(vector) for vector in non_zero_vectors)

    # 使用np.pad函数将每个向量用pad_value补齐到最大长度
    padded_vectors = [np.pad(vector, (0, max_length - len(vector)), 'constant', constant_values=(pad_value, pad_value)) for vector in non_zero_vectors]

    # print("Original Vectors:")
    # print(non_zero_vectors)
    # print("Padded Vectors:")
    # print(padded_vectors)

    unmasked_only_matrix = np.array(padded_vectors)     #unmasked-only matrix
    # print(unmasked_only_matrix)

    return unmasked_only_matrix, mask_positions, masked_matrix, condition2       



# tmp = np.array(tmp).squeeze(1)
# print(tmp.shape)#(20, 22)
# unmasked_only_matrix,mask_positions, masked_matrix = random_mask_with_position(tmp, 0.3) #mask_value=-1, pad_value=-2
# print(mask_positions)
# print(masked_matrix)
# print(unmasked_only_matrix)


#-------------------embedding---------------------------------------
# token_emb = AutoDiscretizationEmbedding2(20, unmasked_only_matrix.shape[1], 
#                                          bin_num=10, 
#                                          bin_alpha=1.0, 
#                                          pad_token_id=-2, 
#                                          mask_token_id=-1)
# pos_emb = nn.Embedding(unmasked_only_matrix.shape[1]+1, 20)

# x = token_emb(torch.unsqueeze(torch.from_numpy(unmasked_only_matrix.astype(np.float32)), 2), output_weight = 0)
# #todo  encoder_position_gene_ids获取方式
# # 创建一个整数张量作为示例
# #encoder_position_gene_ids应该由mask函数得到，T、S没有gene embedding
# #也可能在load.py中
# encoder_position_gene_ids = np.ones_like(unmasked_only_matrix, dtype=np.int64)
# print(encoder_position_gene_ids)
# position_emb = pos_emb(torch.from_numpy(encoder_position_gene_ids))
# print(x)
# print(x.shape)#([20, 4, 20])
# x += position_emb
# print(x)
# print(x.shape)#([20, 4, 20])

def get_unmasked_only_matrix(data):
    print('----------------------------------------------------------------')
    t_marix,tlist,slist = Hierarchical_Bayesian_downsampling(data)
    # print('t_marix',t_marix)
    # print('tlist',tlist)
    # print('slist',slist)

    adata = sc.AnnData(t_marix)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    gexpr_feature = pd.DataFrame(adata.X,index=adata.obs_names,columns=adata.var_names)
    # print('gexpr_feature\n',gexpr_feature)
    tmp = []
    for i in tqdm(range(gexpr_feature.shape[0])):
        # 用totalcount = gexpr_feature.iloc[i,:].sum()计算T，无法计算S
        # 直接对T、S进行log变换，不确定是否能准确表示表达量
        # 暂时直接用log(1+x)
        totalcount = np.log1p(tlist[i])
        sourcecount = np.log1p(slist[i])
        tmpdata = (gexpr_feature.iloc[i,:]).tolist()
        pretrain_gene_x = torch.tensor(tmpdata+[totalcount,sourcecount]).unsqueeze(0)
        # data_gene_ids = torch.arange(gexpr_feature.shape[1]+2, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)
        tmp.append(pretrain_gene_x)
    # print('tmp\n',tmp)

    tmp = np.array(tmp).squeeze(1)
    print(tmp.shape)#(20, 22)
    unmasked_only_matrix,mask_positions, masked_matrix, not_zero_position = random_mask_with_position(tmp, 0.3) #mask_value=-1, pad_value=-2
    # print(mask_positions)
    # print(masked_matrix)
    # print(unmasked_only_matrix)
    print('unmasked_only_matrix.shape',unmasked_only_matrix.shape)
    return unmasked_only_matrix,mask_positions,masked_matrix,not_zero_position


#?---------------得到unmasked_only_matrix-------------------

def get_emb(unmasked_only_matrix):
    token_emb = AutoDiscretizationEmbedding2(20, unmasked_only_matrix.shape[1], 
                                         bin_num=10, 
                                         bin_alpha=1.0, 
                                         pad_token_id=-2, 
                                         mask_token_id=-1)
    pos_emb = nn.Embedding(unmasked_only_matrix.shape[1]+1, 20)
    x = token_emb(torch.unsqueeze(unmasked_only_matrix.to(dtype=torch.float32), 2), output_weight = 0)
    #todo  encoder_position_gene_ids获取方式
    # 创建一个整数张量作为示例
    #encoder_position_gene_ids应该由mask函数得到，T、S没有gene embedding
    #也可能在load.py中
    encoder_position_gene_ids = np.ones_like(unmasked_only_matrix, dtype=np.int64)
    # print(encoder_position_gene_ids)
    position_emb = pos_emb(torch.from_numpy(encoder_position_gene_ids))
    # print(x)
    # print(x.shape)#([x, 4, 20])
    x += position_emb
    # print(x)
    print('x.shape',x.shape)#([x, 4, 20])

# test(matrix)

# _,_,_,a=random_mask_with_position(matrix,0.3)
# print(a)