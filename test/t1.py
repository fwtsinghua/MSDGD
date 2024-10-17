import pandas as pd

# 假设CSV文件的路径为 "data.csv"
# 读取CSV文件
df = pd.read_csv("data.csv")

for num_columns in [40]:
    for num_categories in [1, 2, 3, 4]:
        # for num_categories_child in [1, 2, 3, 4]:
            # 筛选满足条件的行
        filtered_df = df[(df["num_categories"] == num_categories) & (df["num_columns"] == num_columns)]
        savepath = f"{num_columns}_{num_categories}.csv"
        # 将筛选后的数据保存到新的CSV文件中
        filtered_df.to_csv(savepath, index=False)

        # 输出筛选后的数据行数，以确认操作结果
        print(filtered_df.shape[0])