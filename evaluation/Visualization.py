
import random
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def visual_densities_kde(csv_file_1 , csv_file_2):
    # 读取两个CSV文件
    # 假设CSV文件中的数据列名为'Value'
    data_1 = pd.read_csv(csv_file_1).iloc[0]
    data_2 = pd.read_csv(csv_file_2).iloc[0]
    # 计算两个数据集的核密度估计
    kde_1 = gaussian_kde(data_1)  # 第1行数据，生成第一个数据集的KDE
    kde_2 = gaussian_kde(data_2)
    # 生成用于绘制的线条范围，这里假设数据的范围在0到最大值之间
    x_min, x_max = min(data_1), max(data_1)
    # 生成线条的x值，这里我们使用1000个点来生成平滑的曲线
    x_values = np.linspace(x_min, x_max, 1000)
    # 计算两个KDE曲线的y值
    y_values_1 = kde_1(x_values)
    y_values_2 = kde_2(x_values)
    # 绘制两个数据集的KDE曲线
    plt.figure(figsize=(10, 6))  # 设置图形的大小
    plt.plot(x_values, y_values_1, label='GroundTruth', color='blue')
    plt.plot(x_values, y_values_2, label='SimDiff-Trans', color='red')
    # 添加标题、标签和图例
    plt.title('Comparison of Multimodal Distributions using KDE')
    plt.xlabel('Time')
    plt.ylabel('Density')
    plt.legend()
    # 显示图形
    plt.show()



# 定义计算经验均值对数似然的函数
def empirical_mean_log_likelihood(samples):
    return np.mean(np.log(samples))


def visual_densities_hist(csv_file_1 , csv_file_2):
    # 读取两个CSV文件
    # 假设CSV文件中的数据列名为'Value'
    data_1 = pd.read_csv(csv_file_1).iloc[0]
    data_2 = pd.read_csv(csv_file_2).iloc[0]
    # 计算两个数据集的经验均值对数似然
    empirical_log_likelihood_1 = empirical_mean_log_likelihood(data_1)
    empirical_log_likelihood_2 = empirical_mean_log_likelihood(data_2)
    print(f"Empirical mean log-likelihood of Groundtruth (first dataset): {empirical_log_likelihood_1}")
    print(f"Empirical mean log-likelihood of Groundtruth (second dataset): {empirical_log_likelihood_2}")
    # 可视化两个数据集的分布
    plt.figure(figsize=(10, 6))  # 设置图形的大小
    # 绘制第一个数据集的直方图
    plt.hist(data_1, bins=50, density=True, alpha=0.7, label='GroundTruth', color='blue', histtype='stepfilled')
    # 绘制第二个数据集的直方图
    plt.hist(data_2, bins=50, density=True, alpha=0.7, label='SimDiff-Trans', color='red', histtype='stepfilled')
    # 添加标题、标签和图例
    plt.title('Comparison of Multimodal Interarrival-Time Distributions')
    plt.xlabel('Time')
    plt.ylabel('Density')
    plt.legend()
    # 显示图形
    plt.show()




def visualize_table_distribution(table,savepath):
    """

    :param table: df.DataFrame
    :return:
    """
    # visualize the generated samples
    for col_name in table.columns:  # 遍历每一列，将列名作为变量名，将列值保存到变量中
        globals()[col_name] = table[col_name]
        random_color = '#' + ''.join(random.choice('0123456789ABCDEF') for _ in range(6))  # 获取随机颜色
        plt.hist(table[col_name], bins=30, alpha=0.5, color=random_color, label=col_name)
    plt.title('Distributions')  # 添加标题和标签
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()  # 添加图例
    plt.savefig(savepath)
    plt.show()  # 显示图形