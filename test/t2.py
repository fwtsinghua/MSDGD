import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def zhexiantu(datapath):
    # 将数据放入DataFrame
    df = pd.read_csv(datapath)
    # 按方法分组并计算每组的列比例差异均值的平均值
    grouped_df = df.groupby("方法").mean()
    # 绘制折线图
    # 修正数据准备方式
    # 这次我们直接使用原始数据，不需要计算平均值，因为每个num_categories只有一个值
    # 我们将使用不同的颜色来区分不同的方法
    # 绘制折线图



    # 选择一个颜色循环，例如viridis plt.cm.plasma.colors
    colors = plt.cm.plasma(np.linspace(0, 1, len(df["方法"].unique())))
    color_idx = 0

    method_list = [
        "Diffusion_MLP_Intern",
        "Diffusion_MLP_Entire",
        "Diffusion_MLP_ParentChild",
        "RamCol_MLP_Intern",
        "RamCol_MLP_Entire",
        "RamCol_MLP_ParentChild",
        "DimTrans_Unet_Intern",
        "DimTrans_Unet_Entire",
        "DimTrans_Unet_ParentChild",
        "RamCol+DimTrans_Unet_Intern",
        "RamCol+DimTrans_Unet_Entire",
        "RamCol+DimTrans_Unet_ParentChild"]

    plt.figure(figsize=(10, 6))

    for method in method_list:
        subset = df[df["方法"] == method]
        plt.plot(subset["num_categories_child"], subset["列比例差异均值"], marker='o', label=method, color=colors[color_idx])
        color_idx += 1

        # 如果颜色用完，重新开始循环
        if color_idx >= len(colors):
            color_idx = 0

    # 添加轴标题和图例
    plt.xlabel("num_categories_child")
    plt.ylabel("average_difference↓")
    plt.title("Different Child Categories on error_rate↓ of Columns")
    plt.legend()
    # 显示图表
    plt.grid(True)
    plt.savefig(datapath.replace(".csv", ".png"))

import os

# 假设文件夹路径为 "folder_path"
folder_path = "./40"  # 请替换为实际文件夹路径

# 获取文件夹中的所有文件名
all_files = os.listdir(folder_path)

for f in all_files:
    if f.endswith(".csv"):
        zhexiantu(os.path.join(folder_path, f))


