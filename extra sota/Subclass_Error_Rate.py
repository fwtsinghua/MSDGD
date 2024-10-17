import numpy as np
import pandas as pd

# 检查子列中的值是否属于其父列的子类别
def is_subclass_of_parent(row, subclass_col, parent_col, parent_child_mapping):
    subclass_value = row[subclass_col]
    parent_value = row[parent_col]
    valid_children = parent_child_mapping[parent_col].get(parent_value, [])
    return subclass_value in valid_children


def subclass_error_rate(df, subclass_mapping, parent_child_mapping, savepath=None):
    """
    计算DataFrame中的子类别错误率，错误的数量是子列中的值不属于其父列的子类别的数量
    :param df: DataFrame对象
    :param subclass_mapping: 字典，指定子列和父列的对应关系
    :param parent_child_mapping: 字典，父类别和对应的子类别的映射
    :return: 错误率
    """

    error_count = 0  # 错误的总数
    total_pairs = 0  # 检查的子父对总数

    # 遍历所有子父列对
    for subclass_col, parent_col in subclass_mapping.items():
        for index, row in df.iterrows():
            # 检查每一对子父值是否正确
            if not is_subclass_of_parent(row, subclass_col, parent_col, parent_child_mapping):
                error_count += 1  # 错误的子父对数量加1
            total_pairs += 1  # 检查的子父对数量加1

    # 计算错误率
    error_rate = error_count / total_pairs if total_pairs > 0 else 0
    with open(savepath, 'w', encoding='utf-8') as f:
        print(f"----------错误率: {error_rate:.2%}, 错误数量: {error_count}----------", file=f)
    print(f"----------错误率: {error_rate:.2%}, 错误数量: {error_count}----------")
    return error_rate, error_count


if __name__ == '__main__':
    # 定义列之间的子类关系映射
    subclass_mapping = {
        "Column 2": "Column 1",
        "Column 4": "Column 3",
    }
    # 定义父类别和对应的子类别
    parent_child_mapping = {
        "Column 1": {
            "a": ['a1', 'a2', 'a3', 'a4'],
            "b": ["b1", "b2"],
        },
        "Column 3": {
            "c": ['c1', 'c2', 'c3', 'c4'],
            "d": ["d1", "d2"],
        },
    }
    size = 100

    # 生成数据
    column1_data = np.random.choice(list(parent_child_mapping["Column 1"].keys()), size=size)
    column2_data = np.array([parent_child_mapping["Column 1"][c][np.random.randint(0, len(parent_child_mapping["Column 1"][c]))] for c in column1_data])
    column3_data = np.random.choice(list(parent_child_mapping["Column 3"].keys()), size=size)
    column4_data = np.array([parent_child_mapping["Column 3"][c][np.random.randint(0, len(parent_child_mapping["Column 3"][c]))] for c in column3_data])

    # 创建DataFrame
    data = {
        'Column 1': column1_data,
        'Column 2': column2_data,
        'Column 3': column3_data,
        'Column 4': column4_data,
    }
    df = pd.DataFrame(data)

    # 计算并打印错误率
    error_rate, error_count = subclass_error_rate(df, subclass_mapping, parent_child_mapping)
    print(f"错误率: {error_rate:.2%}, 错误数量: {error_count}")

# from Subclass_Error_Rate import subclass_error_rate
# error_rate, error_count = subclass_error_rate(samples, subclass_mapping, parent_child_mapping, savepath=results_path+'/subclass_error_rate.txt')
