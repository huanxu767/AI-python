import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def get_X(df):#读取特征
#     """
#     use concat to add intersect feature to avoid side effect
#     not efficient for big dataset though
#     """
    ones = pd.DataFrame({'ones': np.ones(len(df))})#ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并
    # #df.as_matrix()改写成 df.values
    return data.iloc[:, :-1].values  # 这个操作返回 ndarray,不是矩阵


def get_y(df):#读取标签
#     '''assume the last column is the target'''
    return np.array(df.iloc[:, -1])#df.iloc[:, -1]是指df的最后一列


def normalize_feature(df):
#     """Applies function along input axis(default 0) of DataFrame."""
    return df.apply(lambda column: (column - column.mean()) / column.std())#特征缩放

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

if __name__ == '__main__':
    data = pd.read_csv('./data/ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
    print(data.head())
    sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2))

    sns.lmplot('exam1', 'exam2', hue='admitted', data=data,
               size=6,
               fit_reg=False,
               scatter_kws={"s": 50}
               )
    plt.show()
    X = get_X(data)
    y = get_y(data)

