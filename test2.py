
import pandas as pd

df = pd.DataFrame(
    {
        "Column 0": ['a','a','b','b','a'],
        "Column 1": ['a1','a2','b1','b2','a1'],
        "Column 2": ['c','c','d','d','c'],
        "Column 3": ['c1','c2','d1','d2','c1'],
    }
)

# 增加一列label，用于标记不同行的数据， 即如果在df中出现一行数据是[a, a1, c, c1]，则label为0, 以此类推
df['label'] = df.groupby(df.columns.tolist()).ngroup()
print(df)



# Step 1: Get the list of all column names in the DataFrame.
column_names = df.columns.tolist()

# Step 2: Group the DataFrame by all of its columns.
grouped = df.groupby(column_names)

# Step 3: Assign a unique group number to each group.
group_numbers = grouped.ngroup()

# Step 4: Add these group numbers as a new column to the DataFrame.
df['label'] = group_numbers