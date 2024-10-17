import torch
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy


def read_table(df, batch_size: int = 32, random_state: int = 0) -> (QuantileTransformer, torch.utils.data.DataLoader):
    """

    :param batch_size:
    :param df_values:
    :param random_state:
    :return:
    """
    # train_raw = pd.read_csv('data/data.csv', skiprows=[0])
    # Transform the numerical attributes.

    label_df = df['label']
    label_tensor = torch.IntTensor(label_df.values)

    rows_df = df[df.columns[:-1]]
    rows_tensor = torch.FloatTensor(rows_df.values)
    num_scaler = QuantileTransformer(output_distribution='normal',
                                     random_state=random_state)  # init the quantile transformation
    num_scaler.fit(rows_tensor)  # fit transformation to numerical attributes
    train_num_scaled = num_scaler.transform(rows_tensor)  # transform numerical attributes
    train_num_torch = torch.FloatTensor(train_num_scaled)  # convert numerical attributes
    train_set = TensorDataset(train_num_torch, label_tensor)  # init tensor dataset
    dataloader = DataLoader(  # Init the data loader
        dataset=train_set,  # training dataset
        batch_size=batch_size,  # training batch size
        num_workers=0,  # number of worker
        drop_last=True,  # drop last batch
        shuffle=True  # shuffle training data
    )
    print(f"训练集的表格尺寸为{train_num_torch.size}")

    return num_scaler, dataloader
