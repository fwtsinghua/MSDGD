import torch
import random

def sdtr_transform2(df_torch, num_columns, subclass_mapping, parent_child_mapping):
    """
    列随机重拍，parents-childs两列一起重拍

    num_columns = 16
    subclass_mapping， 每一父列可选的类别和子列可选类别，父列'Column 0'可选'a','b', 其子列可选'a0','a1','b0','b1', 例如 = {'Column 0': {'a': ['a0', 'a1'], 'b': ['b0', 'b1']}, 'Column 2': {'c': ['c0', 'c1'], 'd': ['d0', 'd1']}, 'Column 4': {'e': ['e0', 'e1'], 'f': ['f0', 'f1']}, 'Column 6': {'g': ['g0', 'g1'], 'h': ['h0', 'h1']}}
    parent_child_mapping {"子列":"父列"}， 例如 {'Column 1': 'Column 0', 'Column 3': 'Column 2', 'Column 5': 'Column 4', 'Column 7': 'Column 6'}
    """

    # 生成列的索引
    rows, cols = df_torch.size()
    repeat = int(num_columns / cols)

    # 生成列的索引，保证在字典parent_child_mapping中定义({"子列":"父列"}) 的父列和其子列在一起
    new_dataset_index = random_column_index(cols, repeat, parent_child_mapping, subclass_mapping)

    # 根据new_dataset_index重拍df表格
    new_dataset_matrix = column_split(df_torch, new_dataset_index, parent_child_mapping, subclass_mapping)

    return new_dataset_matrix, new_dataset_index

