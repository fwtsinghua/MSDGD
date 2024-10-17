import random
import numpy as np
import torch



def random_column_index(table_column_length, image_size=256):
    """
    生成表格列的重复索引 论文中的$I^M$
    :param table_column_length: 表格列数量
    :param image_size: 生成图片的长宽，default为256x256
    :return: 列的索引
    """
    repeat_count = image_size // table_column_length  # 重复次数$r$
    extra_count = image_size % table_column_length  # 额外的多余项个数$rem$

    # 随机排列列属性数值并生成列表
    initial_list = list(range(table_column_length))  # 生成一个0到table_column_length-1的列表

    # 重复17次，前16次所有数值+第17次列表的前2个数值
    new_list = initial_list.copy()
    for i in range(repeat_count - 1):  # 除去初始的那一组列表
        random.shuffle(initial_list)
        new_list += initial_list
        if i == repeat_count - 1 - 1:  # 最后一次循环
            random.shuffle(initial_list)
            new_list += initial_list[:extra_count]  # 最后一次，增加额外的多余项
    # column_index = torch.tensor(new_list)  # 将列表转换为PyTorch张量
    # print(f"生成表格列的重复索引:{new_list}")

    return new_list


def column_split(A, column_index):
    """
    拆分阶段：将矩阵A按照列索引column_index进行拆分，得到一个新的矩阵B
    A: 4x4的矩阵
    column_index: 列索引数组，长度为8
    return: 拆分后的B矩阵，4x8的矩阵
    """
    B = torch.zeros((A.shape[0], len(column_index)))  # 创建一个新的0矩阵B
    for i, idx in enumerate(column_index):  # 根据列索引将A的每一列复制到B中的相应列
        B[:, i] = A[:, idx]
    return B


def sdtr_transform(encoded_df, M):
    """

    :param encoded_df:  [100,12]
    :param M: [36]
    :return:
    """

    # 生成列的索引
    rows, cols = encoded_df.size()
    new_dataset_index = random_column_index(cols, image_size=M)

    # 根据column_index重新组合矩阵
    new_dataset_matrix = column_split(encoded_df, new_dataset_index)

    return new_dataset_matrix, new_dataset_index


def column_merge(B, column_index):
    """
    合并阶段：将矩阵B按照列索引column_index进行合并，得到一个新的矩阵A
    :param B:
    :param column_index:
    :return:
    """
    # 创建一个字典，用于存储具有相同索引的列序号
    index_to_column_indices = {idx: [] for idx in set(column_index)}

    # 遍历索引张量的元素和索引
    for i, idx in enumerate(column_index):
        index_to_column_indices[idx].append(i)

    # # 创建一个字典，用于存储具有相同索引的列序号
    # index_to_column_indices = {}
    #
    # # 遍历索引张量的元素和索引
    # for i, idx in enumerate(column_index):
    #     # 如果索引不存在于字典中，则将其添加为键，并初始化为一个列表
    #     if idx not in index_to_column_indices:
    #         index_to_column_indices[idx] = [i]
    #     # 否则，将列序号添加到相应的索引键的列表中
    #     else:
    #         index_to_column_indices[idx].append(i)

    rows, cols = B.shape[0], int(max(column_index)) + 1  # 1000x12  M = int(max(column_index)) + 1

    A = np.zeros((rows, cols), dtype=np.float32)  # 建了一个零矩阵A和

    # 遍历所有行
    for i in range(rows):
        # 每一行，遍历所有列
        for j in range(cols):
            # B中对应j列的列表
            index_j = index_to_column_indices[j]
            # 从列表中随机选择一个数
            random_number = random.choice(index_j)
            # 这个数代表B矩阵中该行的哪个列
            A[i][j] = B[i][random_number]

    return A


def sdtr_inverse_transform(samples_non_inverse, new_dataset_index):
    """

    :param samples_non_inverse: [1000x36]
    :param new_dataset_index:  [36]
    :return:
    """
    samples_non_onehot = column_merge(samples_non_inverse, new_dataset_index)
    return samples_non_onehot


def spatial_dimensional_transformation(A: torch.Tensor, part_size: int) -> torch.Tensor:
    """
    Split a tensor into parts of given size and then combine them into a new tensor.

    Args:
    - A (torch.Tensor): Input tensor to split and combine.
    - part_size (int): Size of each part.

    Returns:
    - result (torch.Tensor): Combined tensor of the parts.
    """
    parts = [A[i:i + part_size, :] for i in range(0, A.size(0), part_size)]
    result = torch.stack(parts)
    return result


def spatial_dimensional_transformation_inverse(result: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct the original tensor from the combined tensor.

    Args:
    - result (torch.Tensor): Combined tensor.

    Returns:
    - A_reconstructed (torch.Tensor): Reconstructed tensor.
    """
    A_reconstructed = torch.cat([result[i] for i in range(result.size(0))], dim=0)
    return A_reconstructed


if __name__ == '__main__':
    # 生成一个大小为768x16的随机矩阵向量 服从标准正态分布的随机矩阵向量，并指定大小为768x16
    dataset = torch.randn(100, 4)
    # table = torch.arange(768*4).view(768,4)
    # table = torch.unsqueeze(torch.arange(0, 16), dim=0).repeat(768, 1)

    dataset_row, dataset_column = dataset.size()
    M = dataset_column * 3 + 0  # 50 M = column numer * repeat times +  reminder  , batch_size in training step

    new_dataset_matrix, new_dataset_index = sdtr_transform(encoded_df=dataset, M=M)
    new_dataset = sdtr_inverse_transform(samples_non_inverse=new_dataset_matrix, new_dataset_index=new_dataset_index)

    table = dataset
    new_table = torch.from_numpy(new_dataset)
    # 比较是否原始表格已经还原
    print(torch.eq(table, new_table))
    # 计算两个矩阵的差值
    difference = table - new_table
    print(difference)
    # 计算矩阵所有元素的和
    sum = torch.sum(difference)
    print(sum)
    print("hello")

    # 假设A是你的Tensor
    A = torch.randn(96, 4)
    # 将A切分成三个部分，每个部分32行数据
    result = spatial_dimensional_transformation(A, 32)
    print("维度变换后的数据大小", result.size())
    # 将 result 张量拆解回原始张量 A
    A_reconstructed = spatial_dimensional_transformation_reverse(result)
    print("分解后的数据大小:", A_reconstructed.size())  # 输出重构后的张量大小为 (96, 4)
