"""
最简单的离散分布的表格
"""


def create_table_1_old(savepath='table1.csv'):
    size = 1000
    df = pd.DataFrame({
        'Column 1': np.random.choice([1, 2, 3, 4], size=size, p=[0.25, 0.25, 0.25, 0.25]),
        'Column 2': np.random.choice([1, 2], size=size, p=[0.5, 0.5]),
        'Column 3': np.random.choice([1, 2, 3, 4], size=size,
                                     p=[0.1, 0.2, 0.3, 0.4]),
        'Column 4': np.random.choice([1, 2], size=size, p=[0.2, 0.8]),
        'Column 5': np.random.choice([1, 2, 3, 4], size=size, p=[0.25, 0.25, 0.25, 0.25]),
        'Column 6': np.random.choice([1, 2], size=size, p=[0.5, 0.5]),
        'Column 7': np.random.choice([1, 2, 3, 4], size=size,
                                     p=[0.1, 0.2, 0.3, 0.4]),
        'Column 8': np.random.choice([1, 2], size=size, p=[0.2, 0.8])
    })
    df = df.astype('category')
    df.to_csv(savepath, index=False)  # index=False表示不保存行索引到文件
    print("表格长宽：", df.shape)
    return df


import numpy as np
import pandas as pd


def create_table_1(size=1000,num_columns=4,  savepath='table.csv'):
    """

    :param size:
    :param num_columns:
    :param savepath:
    :return: pd.DataFrame
    """
    num_columns = num_columns*4
    cols = {}
    for i in range(num_columns):
        if i % 4 == 0:
            cols[f'Column {i+1}'] = np.random.choice(['a','b'], size=size)
        elif i % 4 == 1:
            cols[f'Column {i+1}'] = np.random.choice(['c','d','e','f'], size=size)
        elif i % 4 == 2:
            cols[f'Column {i+1}'] = np.random.choice(['g','h','i','j','k','l','m','n'], size=size)
        else:
            cols[f'Column {i+1}'] = np.random.choice(['o','p','q','r','s','t','u','v',
                                                      'oa','pa','qa','ra','sa','ta','ua','va',], size=size)

    df = pd.DataFrame(cols)
    # df = df.astype('category')
    df.to_csv(savepath, index=False)  # index=False表示不保存行索引到文件
    print("表格长宽：", df.shape)
    return df



import pandas as pd
import numpy as np
from itertools import product
from string import ascii_lowercase


def create_categories(num_categories):
    if num_categories <= 26:
        return list(ascii_lowercase)[:num_categories]
    elif num_categories <= 702:
        single_letters = list(ascii_lowercase)
        double_letters = [''.join(pair) for pair in product(ascii_lowercase, repeat=2)]
        return single_letters + double_letters[:num_categories - 26]
    else:
        raise ValueError("Number of categories must be less than or equal to 702")


def create_table_1_num_categories(num_rows, num_columns, num_categories, savepath='table.csv'):
    # Generate category labels
    categories = create_categories(num_categories*num_columns)

    # Ensure that we have enough unique categories to fill the columns without repetition
    # if num_categories < num_columns:
    #     raise ValueError("Number of categories must be greater than or equal to the number of columns")

    # Create the table
    table_data = {}
    for col in range(num_columns):
        # Choose random categories without replacement
        choice_ready = categories[col*num_categories:(col+1)*num_categories]
        chosen_categories = np.random.choice(choice_ready, size=num_rows, replace=True)
        table_data[f"Column {col + 1}"] = chosen_categories

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Save to CSV
    df.to_csv(savepath, index=False)

    return df



if __name__ == '__main__':
    create_table_1(3)
    # Example usage:
    df = create_table_1_num_categories(num_rows=10, num_columns=2, num_categories=2)
    print(df)