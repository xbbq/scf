import numpy as np

# N = 19264    #基因总数
# C = 1000     #细胞总数

# random_matrix = np.random.rand(C, N)

# # 生成C个bernoulli(0.5)分布的随机样本
# bernoulli_samples = np.random.binomial(1, 0.5, size=C).tolist()
# # 生成C个 β（2,2）分布的随机数
# b_list = np.random.beta(2, 2, C).tolist()

# for i in range(0, C):
#     if bernoulli_samples[i] == 1:
#         for j in range(0, N):
#             random_matrix[i][j] = np.random.binomial(random_matrix[i][j], b_list[i]).tolist()[0]

def random_mask_with_position(matrix, mask_ratio, mask_value=-1):
    """
    对矩阵的每一行元素进行随机遮盖，并返回两个矩阵：一个是原始值和遮盖标记的矩阵，另一个是被遮盖后的矩阵。
    非零值遮盖概率为零值遮盖概率的十倍。

    参数:
    matrix: 输入的矩阵
    mask_ratio: 遮盖比例，应该在0到1之间
    mask_value: 用于遮盖的值

    返回:
    包含原始值和遮盖标记的矩阵
    被遮盖后的矩阵
    """
    # 确保mask_ratio在合理的范围内
    if mask_ratio < 0 or mask_ratio > 1:
        raise ValueError("mask_ratio should be between 0 and 1")

    # 复制矩阵以避免修改原始数据
    masked_matrix = matrix.copy()
    mask_positions = np.zeros_like(matrix)

    # 遍历每一行
    for i in range(matrix.shape[0]):
        # 分离非零元素和零元素
        non_zero_indices = np.where(matrix[i, :] != 0)[0]
        zero_indices = np.where(matrix[i, :] == 0)[0]
        # print('non_zero_indices',non_zero_indices)

        # 确保非零和零元素的索引数量至少为2
        if len(non_zero_indices) < 2:
            non_zero_indices = np.array([0, 1])
        if len(zero_indices) < 2:
            zero_indices = np.array([0, 1])

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

    return mask_positions, masked_matrix

# 示例使用

# 重新生成一个10x10的零矩阵
matrix = np.zeros((20, 20), dtype=int)

# 随机选择10个位置设置为非零
np.random.seed(0)  # 设置随机种子以获得可重复的结果
for _ in range(40):
    i, j = np.random.randint(0, 20, size=2)
    matrix[i, j] = np.random.randint(10, 100)  # 随机选择一个1到9之间的整数
# matrix = np.random.uniform(-10, 400, size=(10, 10))

mask_ratio = 0.3
mask_value = -1
mask_positions, masked_matrix = random_mask_with_position(matrix, mask_ratio)
print("Original Matrix:")
print(matrix)
print("Mask Positions Matrix:")
print(mask_positions)
print("Masked Matrix with Position:")
print(masked_matrix)


# 使用 where 函数生成一个布尔数组，其中 True 对应于非零和非-1的元素
condition = (masked_matrix != 0) & (masked_matrix != -1)

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
        if element != 0 and element != -1:
            non_zero_vector.append(element)
    #用pad标记-2占位
    if len(non_zero_vector) == 0:
        non_zero_vector.append(-2)
    # 将非零向量添加到列表中
    non_zero_vectors.append(non_zero_vector)

print("Original Matrix:")
print(masked_matrix)
print("New Vectors with Non-Zero Values:")
print(non_zero_vectors)
# 找到向量的最大长度
max_length = max(len(vector) for vector in non_zero_vectors)

# 使用np.pad函数将每个向量用-2补齐到最大长度
padded_vectors = [np.pad(vector, (0, max_length - len(vector)), 'constant', constant_values=(-2, -2)) for vector in non_zero_vectors]

print("Original Vectors:")
print(non_zero_vectors)
print("Padded Vectors:")
print(padded_vectors)

matrix = np.array(padded_vectors)
print(matrix)




























