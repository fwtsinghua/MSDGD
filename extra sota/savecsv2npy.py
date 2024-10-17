import numpy as np
import os


def csv_to_npy(csv_path, npy_path, header_path, delimiter=',', dtype=np.float32):
    """
    读取CSV文件并保存为NumPy数组的.npy文件，同时将列名保存到单独的文件中。

    参数:
    - csv_path: CSV文件的路径。
    - npy_path: .npy文件的保存路径。
    - header_path: 列名列的保存路径。
    - delimiter: CSV文件的分隔符，默认为逗号(,)。
    - dtype: 读取数据时的数据类型，默认为np.float32。
    """
    # 读取CSV文件，跳过第一行（列名）
    data = np.genfromtxt(csv_path, delimiter=delimiter, skip_header=1, dtype=dtype)

    # 保存数据为.npy文件
    np.save(npy_path, data)

    # 读取第一行作为列名，并保存到单独的文件
    with open(csv_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()  # 读取第一行作为列名
    with open(header_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')

    print(f'数据已从 {csv_path} 保存到 {npy_path}，列名保存到 {header_path}')


def npy_to_csv(npy_path, csv_path, header_path, delimiter=',', fmt='%s'):
    """
    从NumPy数组的.npy文件读取数据并保存为CSV文件，同时从单独的文件中读取列名。

    参数:
    - npy_path: .npy文件的路径。
    - csv_path: CSV文件的保存路径。
    - header_path: 列名列的读取路径。
    - delimiter: CSV文件的分隔符，默认为逗号(,)。
    - fmt: 输出数据的格式，默认为字符串('%s')。
    """
    # 从.npy文件读取数据
    data = np.load(npy_path)

    # 从单独的文件中读取列名
    with open(header_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()

    # 将列名和数据保存为CSV文件
    np.savetxt(csv_path, data, delimiter=delimiter, header=header, fmt=fmt)

    print(f'数据已从 {npy_path} 保存到 {csv_path}，列名从 {header_path} 读取')


# 示例使用
csv_file_path = 'Emergency_CallCenter_NIM-VL.csv'
npy_file_path = 'samples.npy'
header_file_path = 'Emergency_CallCenter_header.txt'

# 将CSV转换为NPY，同时保存列名到单独的文件
# csv_to_npy(csv_file_path, npy_path=npy_file_path, header_path=header_file_path)

# 将NPY转换回CSV，同时从单独的文件中读取列名
npy_to_csv(npy_path=npy_file_path, csv_path=csv_file_path, header_path=header_file_path)