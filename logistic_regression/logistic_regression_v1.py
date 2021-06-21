import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import scipy.optimize as opt

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

def cost(theta, X, y):
    ''' cost fn is -l(theta) for you to minimize'''
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

# X @ theta与X.dot(theta)等价

def gradient(theta, X, y):
#     '''just 1 batch gradient'''
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)

def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)

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

    theta = theta = np.zeros(3)  # X(m*n) so theta is n*1

    cost1 = cost(theta, X, y)
    print('cost1=',cost1)

    res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='Newton-CG', jac=gradient)
    print(res)

    final_theta = res.x
    y_pred = predict(X, final_theta)

    print(classification_report(y, y_pred))

    print(res.x)  # this is final theta
    coef = -(res.x / res.x[2])  # find the equation
    print(coef)

    x = np.arange(130, step=0.1)
    y = coef[0] + coef[1] * x
    data.describe()  # find the range of x and y

    sns.set(context="notebook", style="ticks", font_scale=1.5)

    sns.lmplot('exam1', 'exam2', hue='admitted', data=data,
               size=6,
               fit_reg=False,
               scatter_kws={"s": 25}
               )

    plt.plot(x, y, 'grey')
    plt.xlim(0, 130)
    plt.ylim(0, 130)
    plt.title('Decision Boundary')
    plt.show()