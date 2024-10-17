import numpy as np
import pandas as pd
import torch
import os
import json
from scripts.RamCol_2  import sdtr_transform2, sdtr_transform2_reverse

from scripts.create_data.TableNumerical import (InternEncoder_Inverse, InternEncoder, EntireEncoder, generate_dataframe,
                                                EntireEncoder_Inverse, ParentChildEncoder, ParentChildEncoder_Inverse)

from model.Table_Random.column_types import infer_column_types
from model.Table_Random.generate_data import generate_data

def run(model_type,data_path, epochs, lr, diffusion_steps):
    """
    不同不同数据集+不同方法，需要
    修改： model_type  +  表格生成代码 + SDTR 方法--构造数据 + SDTR 方法--解析数据
    """

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(0)  # 指定GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = epochs
    lr = lr
    results_path = 'results'

    """
    =========================================================================
     0. Data  
    =========================================================================
    """
    if "Table2" in experiments:
        with open(data_path.replace('.csv', '_subclass_mapping.json')) as f:
            subclass_mapping = json.load(f)
        with open(data_path.replace('.csv', '_parent_child_mapping.json')) as f:
            parent_child_mapping = json.load(f)

    """
    1. 生成数据
    """

    if "Random" in model_type:
        infer_column_types(csv_file=data_path, output_json=data_path.replace('.csv', '_types.json'))
        generate_data(csv_file=data_path, row_num=1000,
                                     type_dict=data_path.replace('.csv', '_types.json'),
                                     output_file=data_path.replace('.csv', '_gen.csv'))

    """
    2 评价数据
    """
    df = pd.read_csv(data_path)  # 读取CSV文件
    samples = pd.read_csv(data_path.replace('.csv', '_gen.csv'))

    from evaluation.Ratio_Difference import ratio_difference
    average_difference = ratio_difference(df, samples, savepath=results_path+'/ratio_difference.txt')

    error_rate, error_count= 99, 99
    if "Table2" in experiments:
        from evaluation.Subclass_Error_Rate import subclass_error_rate
        error_rate, error_count = subclass_error_rate(samples, subclass_mapping, parent_child_mapping, savepath=results_path+'/subclass_error_rate.txt')

    from evaluation.SVDevaluation import svd_evaluation
    Column_Shapes, Column_Pair_Trends = svd_evaluation(real_data=df, synthetic_data=samples, savepath=results_path+'/svd_evaluation')


    return average_difference, Column_Shapes, Column_Pair_Trends, error_rate, error_count

    # print("| 方法    |epoch        | 列比例差异均值  | fidelity Column | fidelity row  |    错误分类率 |   错误数量 |")
    # print("| --------|------- | --------------- | --------------- | -------------- |-------------- |-------------- |")
    # if "Table1" in experiments:
    #     print(f"| {model_type} | {epochs}|{average_difference:.4f} | {Column_Shapes:.4f} | {Column_Pair_Trends:.4f} | - | - |")
    # if "Table2" in experiments:
    #     print(f"| {model_type} |{epochs}| {average_difference:.4f} | {Column_Shapes:.4f} | {Column_Pair_Trends:.4f} | {error_rate:.4f} | {error_count} |")


"""
num_rows = 1000
num_columns = 50 # Table2 : 生成2倍num_columns的列数   ;  Table1 : 生成num_columns列
num_categories = 100
# num_categories_child = 4
experiments = 'Table1'
"""
num_rows = 1000
num_columns = 20 # Table2 : 生成2倍num_columns的列数   ;  Table1 : 生成num_columns列
num_categories = 8
num_categories_child = 4
experiments = 'Table2'
data_path = f"data/{experiments}/{num_rows}x{num_columns}_{num_categories}_{num_categories_child}.csv"

epochs = 10
lr = 1e-3
diffusion_steps=700
model_type = 'Table_Random'
print("| 方法 | num_rows|num_columns |num_categories|num_categories_child  |epochs|  diffusion_steps      | 列比例差异均值  | fidelity Column | fidelity row  |    错误分类率 |   错误数量 |")
print("| --------|------- | --------------- | --------------- | -------------- |-------------- |------ |------ |------ |------ |------ |-------------- |")

average_difference, Column_Shapes, Column_Pair_Trends, error_rate, error_count=run(model_type,data_path, epochs, lr,diffusion_steps)
print(f"| {model_type} |{num_rows}|{num_columns}|{num_categories}|{num_categories_child}|{epochs}|{diffusion_steps}| {average_difference:.4f} | {Column_Shapes:.4f} | {Column_Pair_Trends:.4f} | {error_rate:.4f} | {error_count} |")

