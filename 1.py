import pandas as pd

# parent_child_mapping ={'Column 0': {'a': ['a0', 'a1'], 'b': ['b0', 'b1'], 'c': ['c0', 'c1'], 'd': ['d0', 'd1']},
#  'Column 2': {'e': ['e0', 'e1'], 'f': ['f0', 'f1'], 'g': ['g0', 'g1'], 'h': ['h0', 'h1']}}
#
# subclass_mapping = {'Column 1': 'Column 0', 'Column 3': 'Column 2'}
#
# df = pd.DataFrame({'Column 0': ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
#       'Column 1': ['a0', 'a1', 'b0', 'b1', 'c0', 'c1', 'd0', 'd1'],
#  'Column 2': ['e', 'e', 'f', 'f', 'g', 'g', 'h', 'h'],
#  'Column 3': ['e0', 'e1', 'f0', 'f1', 'g0', 'g1', 'h0', 'h1']})


import pandas as pd



def generate_dataframe(parent_child_mapping, subclass_mapping):
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
import pandas as pd

def generate_dataframe(parent_child_mapping, subclass_mapping):
    # 初始化一个空列表来存储数据帧的行
    rows = []

    # 遍历`subclass_mapping`中的每一对（子类列，父类列）
    for subclass_col, parent_col in subclass_mapping.items():
        # 获取父值到其子元素的映射
        parent_to_children = parent_child_mapping[parent_col]

        # 生成所有可能的父值与子值组合
        for parent_value, children_list in parent_to_children.items():
            for child_value in children_list:
                # 创建一行，其中父值位于父类列中，子值位于子类列中
                row = {parent_col: parent_value, subclass_col: child_value}
                rows.append(row)

    # 将行列表转换为数据帧
    df = pd.DataFrame(rows)

    return df

parent_child_mapping = {
    'Column 0': {'a': ['a0', 'a1'], 'b': ['b0', 'b1'], 'c': ['c0', 'c1'], 'd': ['d0', 'd1']},
    'Column 2': {'e': ['e0', 'e1'], 'f': ['f0', 'f1'], 'g': ['g0', 'g1'], 'h': ['h0', 'h1']}
}

subclass_mapping = {'Column 1': 'Column 0', 'Column 3': 'Column 2'}

df = generate_dataframe(parent_child_mapping, subclass_mapping)

print(df)