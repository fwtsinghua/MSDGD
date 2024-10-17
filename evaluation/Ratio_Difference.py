

def ratio_difference(df, samples, savepath=None):
    mean_difference_list = []
    with open(savepath, 'w', encoding='utf-8') as f:
        # 比较每列的分类比例是否与原始设定的比例相同
        for i in range(len(df.columns)):
            column_name = df.columns[i]

            freq_column1 = df[column_name].value_counts(normalize=True).sort_index().tolist()  # 原始数据的频率
            freq_column1_sample = samples[column_name].value_counts(normalize=True).sort_index().tolist()
            # 计算对应元素之间的差并取绝对值
            differences = [abs(x - y) for x, y in zip(freq_column1, freq_column1_sample)]

            # 计算差的均值
            mean_difference = sum(differences) / len(differences)

            # 将打印输出重定向到文件
            print(f"差的均值：{mean_difference}", file=f)
            print(f"{column_name} 原始数据的频率：{freq_column1}", file=f)
            print(f"{column_name} 生成数据的频率：{freq_column1_sample}", file=f)


            print(f"差的均值：{mean_difference}")
            print(f"{column_name} 原始数据的频率：{freq_column1}")
            print(f"{column_name} 生成数据的频率：{freq_column1_sample}")
            mean_difference_list.append(mean_difference)
            
        results = sum(mean_difference_list) / len(mean_difference_list)
        print(f"平均差的均值：{results}", file=f)
        print(f"------------平均差的均值：{results}------------")
    return results