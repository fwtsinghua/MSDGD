import pandas as pd
import numpy as np
import random

import torch


def shuffle_pairs(parent_child_mapping,subclass_mapping):
    # Extract unique parent columns
    parents = list(set(parent_child_mapping.keys()))
    # Shuffle the parents
    random.shuffle(parents)
    # Create a new mapping based on shuffled parents
    new_mapping = {}
    for parent in parents:
        # Find all child columns for this parent
        children = [child for child, par in subclass_mapping.items() if par == parent]
        for child in children:
            new_mapping[child] = parent
    return new_mapping

def sdtr_transform2(df_torch, repeat, subclass_mapping, parent_child_mapping):
    # Function to shuffle parent-child pairs
    # Create an empty DataFrame to concatenate new matrices
    concatenated_matrix = pd.DataFrame()
    # Create an empty list to store new indices
    concatenated_indices = []
    # 从第一列开始判断df_torch是否在subclass_mapping中，如果不在，则直接添加到concatenated_matrix中，否则，父子列只进行一次进行下面步骤
    swapped_mapping = {key:value for value,key in subclass_mapping.items()}
    for col in df_torch.columns:
        if col not in subclass_mapping.keys() and col not in subclass_mapping.values():
            concatenated_matrix[col] = df_torch[col]
            concatenated_indices.append(df_torch.columns.get_loc(col))
        else:
            if col in swapped_mapping.keys():  # key都是父列，不会有子列
                # Repeat the shuffling and concatenation process
                for _ in range(repeat):
                    # Shuffle the parent-child pairs
                    shuffled_mapping = shuffle_pairs(parent_child_mapping,subclass_mapping)

                    # Create a list of column indices based on the shuffled pairs
                    new_dataset_index = []
                    for parent in shuffled_mapping.values():
                        new_dataset_index.append(df_torch.columns.get_loc(parent))
                        child = list(shuffled_mapping.keys())[list(shuffled_mapping.values()).index(parent)]
                        new_dataset_index.append(df_torch.columns.get_loc(child))

                    # Use the new index to reorder the dataframe columns
                    new_dataset_matrix = df_torch.iloc[:, new_dataset_index].copy()

                    # Concatenate the new matrix to the right of the previous matrix
                    concatenated_matrix = pd.concat([concatenated_matrix, new_dataset_matrix], axis=1)
                    # Append the new indices to the concatenated_indices list
                    concatenated_indices.extend(new_dataset_index)


    concatenated_tensor = torch.from_numpy(concatenated_matrix.to_numpy())
    return concatenated_tensor, concatenated_indices



def column_merge(B, column_index):
    """
    合并阶段：将矩阵B按照列索引column_index进行合并，得到一个新的矩阵A
    :param B: tensor, 1000x12
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


def sdtr_transform2_reverse(samples_non_inverse, new_dataset_index):
    """

    :param samples_non_inverse: [1000x36]
    :param new_dataset_index:  [36]
    :return:
    """
    samples_non_onehot = column_merge(samples_non_inverse, new_dataset_index)
    return samples_non_onehot.astype(int)


#
# def sdtr_transform2_reverse(new_dataset_matrix, new_dataset_index, subclass_mapping, parent_child_mapping):
#     # Create the original column order based on the parent_child_mapping
#     original_order = []
#     for parent in sorted(subclass_mapping.values()):
#         # Append parent column index
#         original_order.append(new_dataset_index.index(new_dataset_matrix.columns.get_loc(parent)))
#         # Append child column index
#         child = list(subclass_mapping.keys())[list(subclass_mapping.values()).index(parent)]
#         original_order.append(new_dataset_index.index(new_dataset_matrix.columns.get_loc(child)))
#
#     # Sort the original_order list based on the values (original positions)
#     sorted_original_order = sorted(range(len(original_order)), key=lambda k: original_order[k])
#
#     # Use the sorted_original_order to reorder the dataframe columns
#     original_dataset_matrix = new_dataset_matrix.iloc[:, sorted_original_order].copy()
#
#     return original_dataset_matrix

# Example usage:
# new_dataset_matrix is your shuffled dataframe
# new_dataset_index is the index that was used for shuffling
# subclass_mapping is your subclass mapping dictionary
# parent_child_mapping is your parent-child mapping dictionary

# original_df = sdtr_transform2_reverse(new_dataset_matrix, new_dataset_index, subclass_mapping, parent_child_mapping)