import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


def InternEncoder_Inverse(df, label_encoder_list):
    """
    ## 解码
    # subclass_string_inverse_transform(df, label_encoder_list)
    from sklearn.preprocessing import LabelEncoder

    :param df:
    :param label_encoder_list:
    :return:
    """
    for i, column_name in enumerate(df.columns):
        le = label_encoder_list[i]
        df[column_name] = le.inverse_transform(df[column_name])

    return df


def InternEncoder(df):
    """
    ##  每一列单独编码
    from sklearn.preprocessing import LabelEncoder
    :param df:
    :return:
    """

    label_encoder_list = []
    for column_name in df.columns:
        le = LabelEncoder()
        df[column_name] = le.fit_transform(df[column_name])  # 从0开始编码
        label_encoder_list.append(le)
    return df, label_encoder_list


def EntireEncoder(train_raw):
    train_raw = train_raw.astype(str)
    for col in train_raw.select_dtypes(include='object').columns:
        train_raw[col] = col + "_" + train_raw[col]

    vocabulary_classes = np.unique(train_raw.values.flatten())
    encoder = LabelEncoder()
    encoder.fit(vocabulary_classes)
    encoded_train_raw = train_raw.apply(encoder.transform)

    vocab_per_column = {}
    for col in encoded_train_raw.columns:
        vocab_per_column[col] = set(encoded_train_raw[col])

    return encoded_train_raw, encoder, vocab_per_column


def EntireEncoder_Inverse(encoded_train_raw, encoder):
    inverse_transform_train_raw = encoded_train_raw.apply(encoder.inverse_transform)
    for col in inverse_transform_train_raw.select_dtypes(include='object').columns:
        inverse_transform_train_raw[col] = inverse_transform_train_raw[col].str.replace(col + "_", '')
    return inverse_transform_train_raw


def ParentChildEncoder(df, parent_child_mapping,subclass_mapping):
    class_to_code = {}
    code = 0
    for parent, children in parent_child_mapping.items():
        for child, grand_children in children.items():
            num_grand_children = len(grand_children)
            child_code = code + num_grand_children // 2  # 子类编码为孙类编码的中间值
            class_to_code[child] = child_code
            for grand_child in grand_children:
                while code == child_code:  # 确保孙类编码与子类编码不相等
                    code += 1
                class_to_code[grand_child] = code
                code += 1
        # class_to_code[parent] = sum(class_to_code[child] for child in children.keys()) // len(children)  # 父类编码为所有子类编码的平均值
    print(class_to_code)  # 打印结果，可注释

    label_encoder_list = []
    for col in df.columns:
        if col in subclass_mapping.keys() or col in subclass_mapping.values():
            df[col] = df[col].map(class_to_code)
        else:
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])
            label_encoder_list.append(label_encoder)
    return df, class_to_code, label_encoder_list


def ParentChildEncoder_Inverse(df, class_to_code, subclass_mapping, label_encoder_list):
    inverse_mapping = {v: k for k, v in class_to_code.items()}
    for index, col in enumerate(df.columns):
        if col in subclass_mapping.keys() or subclass_mapping.values():
            df[col] = df[col].map(inverse_mapping)
        else:
            df[col] = label_encoder_list[index].inverse_transform(df[col])
    return df

import pandas as pd

def generate_dataframe(parent_child_mapping, subclass_mapping,parent_child_encoder):
    """
    This code defines a function generate_dataframe() that takes parent_child_mapping and subclass_mapping as inputs and returns the desired DataFrame.
    It first initializes an empty list rows to hold the individual rows of the DataFrame.
    Then, it iterates over each pair of (subclass column, parent column) in subclass_mapping.
    For each pair, it retrieves the mapping of parent values to their child elements from parent_child_mapping.
    Next, it generates all possible combinations of parent and child values, creating a dictionary for each combination representing a single row in the DataFrame.
    Finally, it appends these dictionaries to the rows list.

    After processing all pairs in subclass_mapping, the function converts the rows list into a Pandas DataFrame and returns it.
    When you run the provided code with your given parent_child_mapping and subclass_mapping, it will generate and print the desired DataFrame df.

    :param parent_child_mapping:  This dictionary maps column names to nested dictionaries,
    where the outer keys are parent columns and the inner keys are values that can appear in those parent columns.
    The corresponding inner values are lists containing child elements associated with each parent value.

    :param subclass_mapping:  This dictionary establishes relationships between pairs of columns,  indicating that one column is a subclass of another.
    :return:
    """
    # 初始化一个空列表来存储数据帧的行
    rows = []
    total_number = 0

    # 遍历`subclass_mapping`中的每一对（子类列，父类列）
    for subclass_col, parent_col in subclass_mapping.items():
        # 获取父值到其子元素的映射
        parent_to_children = parent_child_mapping[parent_col]

        # 生成所有可能的父值与子值组合
        for parent_value, children_list in parent_to_children.items():
            total_number += len(children_list)+1
            for child_value in children_list:
                # 创建一行，其中父值位于父类列中，子值位于子类列中
                row = {parent_col: parent_value, subclass_col: child_value}
                rows.append(row)

    # 将行列表转换为数据帧
    df = pd.DataFrame(rows)


    for col in df.columns:
        if col in subclass_mapping.keys() or subclass_mapping.values():
            df[col] = df[col].map(parent_child_encoder)
    df = df.fillna(0)

    return torch.tensor(df.values).float()/total_number