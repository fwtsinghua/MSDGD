import numpy as np


# 父类子类映射关系
parent_child_mapping = {
    "a": ['a1', 'a2', 'a3', 'a4'],
    "b": ["b1", "b2"],
    "c": ['c1', 'c2', 'c3', 'c4'],
    "d": ["d1", "d2"],
    "e": ['e1', 'e2', 'e3', 'e4'],
    "f": ["f1", "f2"]
}


# subclass_mapping ={'Column 2': 'Column 1',
#  'Column 4': 'Column 3',
#  'Column 6': 'Column 5',
#  'Column 8': 'Column 7'}
#
# parent_child_mapping = {'Column 1': {'a': ['a2', 'a3', 'a4', 'a5'], 'b': ['b2', 'b3']},
#  'Column 3': {'c': ['c2', 'c3', 'c4', 'c5'], 'd': ['d2', 'd3']},
#  'Column 5': {'e': ['e2', 'e3', 'e4', 'e5'], 'f': ['f2', 'f3']},
#  'Column 7': {'g': ['g2', 'g3', 'g4', 'g5'], 'h': ['h2', 'h3']}}





def table_numerical(parent_child_mapping=None):
    # 生成类别的数字编码
    class_to_code = {}
    code = 1
    for parent, children in parent_child_mapping.items():
        num_children = len(children)
        parent_code = code + num_children // 2  # 父类编码为子类编码的中间值
        class_to_code[parent] = parent_code
        for child in children:
            while code == parent_code:  # 确保子类编码与父类编码不相等
                code += 1
            class_to_code[child] = code
            code += 1
    print(class_to_code)
    return class_to_code


class_to_code = table_numerical(parent_child_mapping)
# 构建相关性矩阵
num_classes = len(class_to_code)
correlation_matrix = np.zeros((num_classes, num_classes))

# 填充相关性矩阵
for parent, children in parent_child_mapping.items():
    parent_code = class_to_code[parent]
    for child in children:
        child_code = class_to_code[child]
        correlation_matrix[parent_code - 1, child_code - 1] = 1

print("相关性矩阵：")
print(correlation_matrix)
