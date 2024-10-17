"""
生成具有列之间关系的数据集
"""

from scripts.create_data.TableNumerical import InternEncoder


def create_table_2_old(savepath, size = 100):
    # 定义列之间的子类关系映射
    subclass_mapping = {
        "Column 2": "Column 1",
        "Column 4": "Column 3",
        "Column 6": "Column 5",
    }
    # 定义父类别和对应的子类别
    parent_child_mapping = {
        "a": ['a1', 'a2', 'a3', 'a4'],
        "b": ["b1", "b2"],
        "c": ['c1', 'c2', 'c3', 'c4'],
        "d": ["d1", "d2"],
        "e": ['e1', 'e2', 'e3', 'e4'],
        "f": ["f1", "f2"]
    }

    # 生成第1列数据，两类分类 a 和 b
    column1 = np.random.choice(['a', 'b'], size=size, p=[0.5, 0.5])
    # 根据第1列生成第2列数据
    column2_a = np.random.choice(['a1', 'a2', 'a3', 'a4'], size=size, p=[0.25, 0.25, 0.25, 0.25])  # 第1列为a时，对应的取值
    column2_b = np.random.choice(['b1', 'b2'], size=size, p=[0.5, 0.5])  # 第1列为b时，对应的取值
    column2 = np.where(column1 == 'a', column2_a, column2_b)  # 根据第1列的取值选择对应的第2列取值

    # 生成第3列数据，两类分类 c 和 d
    column3 = np.random.choice(['c', 'd'], size=size, p=[0.2, 0.8])
    # 根据第3列生成第4列数据
    column4_c = np.random.choice(['c1', 'c2', 'c3', 'c4'], size=size, p=[0.25, 0.25, 0.25, 0.25])  # 第3列为c时，对应的取值
    column4_d = np.random.choice(['d1', 'd2'], size=size, p=[0.5, 0.5])  # 第3列为d时，对应的取值
    column4 = np.where(column3 == 'c', column4_c, column4_d)  # 根据第3列的取值选择对应的第4列取值

    column5 = np.random.choice(['e', 'f'], size=size, p=[0.2, 0.8])
    column6_e = np.random.choice(['e1', 'e2', 'e3', 'e4'], size=size, p=[0.25, 0.25, 0.25, 0.25])  # 第3列为c时，对应的取值
    column6_f = np.random.choice(['f1', 'f2'], size=size, p=[0.5, 0.5])  # 第3列为d时，对应的取值
    column6 = np.where(column3 == 'e', column4_c, column4_d)  # 根据第3列的取值选择对应的第4列取值

    # 创建DataFrame
    data = {
        'Column 1': column1,
        'Column 2': column2,
        'Column 3': column3,
        'Column 4': column4,
        'Column 5': column5,
        'Column 6': column6,
    }
    df = pd.DataFrame(data)
    print("表格长宽：", df.shape)
    label_encoder_list = InternEncoder(df)

    df = df.astype('category')
    df.to_csv(savepath, index=False)

    return df, label_encoder_list, subclass_mapping, parent_child_mapping


import numpy as np
import pandas as pd
import json


import numpy as np
import pandas as pd
import json
def create_table_2(num_rows=1000, num_columns=4, num_categories=4, num_categories_child=2, savepath="data.csv"):
    subclass_mapping, parent_child_mapping = generate_mappings(num_columns, num_categories, num_categories_child)
    with open(savepath.replace('.csv', '_subclass_mapping.json'), 'w') as f:
        json.dump(subclass_mapping, f)
    with open(savepath.replace('.csv', '_parent_child_mapping.json'), 'w') as f:
        json.dump(parent_child_mapping, f)

    data = {}
    for subclass_col, parent_col in subclass_mapping.items():
        parent_values = parent_child_mapping[parent_col].keys()
        data[parent_col] = np.random.choice(list(parent_values), size=num_rows)
        child_values = []
        # 生成子列数据，子列的值从父列映射而来
        for parent_value in data[parent_col]:
            child_value = np.random.choice(parent_child_mapping[parent_col][parent_value])
            child_values.append(child_value)
        data[subclass_col] = child_values

    df = pd.DataFrame(data)
    print(f"表格长宽：{df.shape}: {num_categories} categories and {num_categories_child} child categories" )
    # df = df.astype('category')
    df.to_csv(savepath, index=False)

    return df, subclass_mapping, parent_child_mapping



import string
import itertools

def generate_mappings(num_columns, num_categories, num_categories_child):
    """
    Generate mappings for parent and child columns.
    Each parent category has num_categories_child subcategories, for a total number of categories.

    :param num_columns: Number of columns in the table.
    :param num_categories: Number of categories for parent columns.
    :param num_categories_child: Number of subcategories for each parent category.
    :return: Tuple containing subclass_mapping and parent_child_mapping.
    """

    # Create a generator for parent column category sequences.
    parent_sequence = itertools.product(string.ascii_lowercase, repeat=num_columns*num_categories // 26 + 1)
    parent_letters = (''.join(s) for s in parent_sequence)

    subclass_mapping = {}
    parent_child_mapping = {}

    # Generate mappings for parent and subclass columns.
    for i in range(num_columns):
        parent_col = f"Column {i * 2}"
        subclass_col = f"Column {i * 2 + 1}"

        subclass_mapping[subclass_col] = parent_col
        parent_child_mapping[parent_col] = {}


        for _ in range(num_categories):
            # Get next parent letter and ensure it's unique.
            parent_letter = next(parent_letters)

            parent_child_mapping[parent_col][parent_letter] = [f"{parent_letter}{k}" for k in range(num_categories_child)]

    return subclass_mapping, parent_child_mapping


def create_table_2_for_paper():
    
    num_rows = 1000
    for num_columns in [2, 4, 8, 12, 16, 20, 25, 40]:
        for num_categories in [1,2,3,4]:
            for num_categories_child in [1,2,3,4]:
                # if num_categories > num_columns:
                data_path = 'data/Table2/'+str(num_rows)+'x'+str(num_columns)+'_'+str(num_categories)+'_'+str(num_categories_child)+'.csv'
                file_name = data_path.replace('data/Table2/', '').replace('.csv', '')
                df, subclass_mapping, parent_child_mapping = create_table_2(num_rows=num_rows, num_columns=num_columns,
                                                                                num_categories=num_categories,num_categories_child=num_categories_child,
                                                                                savepath=data_path)  # 生成2倍num_columns的列数



if __name__ == '__main__':
    """
    subclass_mapping =
    {'Column 2': 'Column 1',
     'Column 4': 'Column 3',
     'Column 6': 'Column 5',
     'Column 8': 'Column 7'}
     
    parent_child_mapping = 
    {'Column 1': {'a': ['a2', 'a3', 'a4', 'a5'], 'b': ['b2', 'b3']},
     'Column 3': {'c': ['c2', 'c3', 'c4', 'c5'], 'd': ['d2', 'd3']},
     'Column 5': {'e': ['e2', 'e3', 'e4', 'e5'], 'f': ['f2', 'f3']},
     'Column 7': {'g': ['g2', 'g3', 'g4', 'g5'], 'h': ['h2', 'h3']}}
    """
    subclass_mapping, parent_child_mapping = generate_mappings(2, 20, num_categories_child=10)  # 双倍
    print("Subclass Mapping:", subclass_mapping)
    print("Parent-Child Mapping:", parent_child_mapping)

    df, subclass_mapping, parent_child_mapping = create_table_2(num_rows=1000, num_columns=2, num_categories=20, num_categories_child=10, savepath="data.csv")

    print("hello")

