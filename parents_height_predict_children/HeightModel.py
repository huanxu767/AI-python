import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Use multiple algorithms to train models

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score



def singleXDemo():
    # Load dataset
    df = pd.read_csv('./data/galton_height_data.csv')
    # 清洗数据------------------------------------------
    # print(df)
    df.drop(df.columns[0],axis=1,inplace=True)
    # print(df)
    # Test 1: Assuming the number of children in the family does not affect the height of the child
    # Drop the 'kids' column
    df.drop(['kids'], axis=1, inplace=True)
    print(df)
    # Label encode the 'gender' column
    g = df['gender'].value_counts()
    print(g)
    print(g.index)
    for i in range(len(g.index)):
        df['gender'].replace(g.index[i],i,inplace=True)
    # 设置 X y变量------------------------------------------
    y = df['height']
    features = ['father','mother','gender']
    X = df[features]

    # Normalise and Standardise Features
    # 标准差标准化 StandardScaler
    # 处理方法：标准化数据减去均值，然后除以标准差，经过处理后数据符合标准正态分布，即均值为0，标准差为1；
    # 转化函数：x = (x - mean) / std；
    # 适用性：适用于本身服从正态分布的数据；
    # Outlier的影响：基本可用于有outlier的情况，但在计算方差和均值时outliers仍然会影响计算。
    X = StandardScaler().fit_transform(X)
    # 极差标准化 / 归一化 MinMaxScaler
    #处理方法：将特征缩放到给定的最小值和最大值之间，也可以将每个特征的最大绝对值转换至单位大小。这种方法是对原始数据的线性变换，将数据归一到[0,1]中间；
    #转换函数：x = (x-min) / (max-min)；
    #适用性：适用于分布范围较稳定的数据，当新数据的加入导致max/min变化，则需重新定义；
    # Outlier 的影响：因为outlier会影响最大值或最小值，因此对outlier非常敏感。
    # X = MinMaxScaler().fit_transform(X)
    # Splitting the dataset into separate train and test sets (60-40)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

    print("Accuracy Scores (Train-Test Split):")

    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    print("Linear Regression:", lin_model.score(X_test, y_test).round(3))

    lasso_model = Lasso()
    lasso_model.fit(X_train, y_train)
    print("Lasso:", lasso_model.score(X_test, y_test).round(3))

    elastic_model = ElasticNet()
    elastic_model.fit(X_train, y_train)
    print("ElasticNet:", elastic_model.score(X_test, y_test).round(3))

    dec_model = DecisionTreeRegressor()
    dec_model.fit(X_train, y_train)
    print("Decision Tree Regressor:", dec_model.score(X_test, y_test).round(3))

    knr_model = KNeighborsRegressor()
    knr_model.fit(X_train, y_train)
    print("K-Neighbors Regressor:", knr_model.score(X_test, y_test).round(3))

    gbr_model = GradientBoostingRegressor()
    gbr_model.fit(X_train, y_train)
    print("Gradient Boosting Regressor:", gbr_model.score(X_test, y_test).round(3))

    rfr_model = RandomForestRegressor()
    rfr_model.fit(X_train, y_train)
    print("Random Forest Regressor:", rfr_model.score(X_test, y_test).round(3))

    # Accuracy Score for 10-fold Cross Validation

    print("Accuracy Scores (10-fold Cross Validation):")

    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    lin_scores = cross_val_score(lin_model, X, y, cv=10)
    print("Linear Regression (CV): %0.2f accuracy with a standard deviation of %0.2f" % (
    lin_scores.mean(), lin_scores.std()))

    lasso_model = Lasso()
    lasso_model.fit(X_train, y_train)
    lasso_scores = cross_val_score(lasso_model, X, y, cv=10)
    print("Lasso: %0.2f accuracy with a standard deviation of %0.2f" % (lasso_scores.mean(), lasso_scores.std()))

    elastic_model = ElasticNet()
    elastic_model.fit(X_train, y_train)
    elastic_scores = cross_val_score(elastic_model, X, y, cv=10)
    print("ElasticNet (CV): %0.2f accuracy with a standard deviation of %0.2f" % (
    elastic_scores.mean(), elastic_scores.std()))

    dec_model = DecisionTreeRegressor()
    dec_model.fit(X_train, y_train)
    dec_scores = cross_val_score(dec_model, X, y, cv=10)
    print("Decision Tree Regressor (CV): %0.2f accuracy with a standard deviation of %0.2f" % (
    dec_scores.mean(), dec_scores.std()))

    knr_model = KNeighborsRegressor()
    knr_model.fit(X_train, y_train)
    knr_scores = cross_val_score(knr_model, X, y, cv=10)
    print("K-Neighbors Regressor (CV): %0.2f accuracy with a standard deviation of %0.2f" % (
    knr_scores.mean(), knr_scores.std()))

    gbr_model = GradientBoostingRegressor()
    gbr_model.fit(X_train, y_train)
    gbr_scores = cross_val_score(gbr_model, X, y, cv=10)
    print("Gradient Boosting Regressor: %0.2f accuracy with a standard deviation of %0.2f" % (
    gbr_scores.mean(), gbr_scores.std()))

    rfr_model = RandomForestRegressor()
    rfr_model.fit(X_train, y_train)
    rfr_scores = cross_val_score(rfr_model, X, y, cv=10)
    print("Random Forest Regressor (CV): %0.2f accuracy with a standard deviation of %0.2f" % (
    rfr_scores.mean(), rfr_scores.std()))

    # Using Linear Regressor-trained model to predict height

    lin_pred = lin_model.predict(X_test)
    ser_lin_pred = pd.Series(np.round(lin_pred, 1))
    ser_lin_pred.name = "Predicted"

    ser_y_test = pd.Series(y_test)
    ser_y_test = ser_y_test.reset_index(drop=True)
    ser_y_test.name = "Actual"

    difference = ser_y_test - ser_lin_pred
    difference.name = 'Actual - Prediction'

    prediction = pd.concat([ser_y_test, ser_lin_pred, difference], axis=1).reset_index(drop=True)
    print(prediction)

# Test 2: Taking into consideration the number of children in the family
def model2HandleData():
    df = pd.read_csv('./data/galton_height_data.csv')
    df.drop(df.columns[0],axis=1,inplace=True)
    g = df['gender'].value_counts()
    for i in range(len(g.index)):
        df['gender'].replace(g.index[i],i,inplace=True)
    return df
def model2DefineXandY(df):
    # 设置 X y变量------------------------------------------
    y = df['height']
    features = ['father','mother','gender','kids']
    X = df[features]
    return X,y
if __name__ == '__main__':
    # 1，处理数据
    df = model2HandleData();
    # 2，设置X y
    X,y = model2DefineXandY(df);
    # 3，归一化 标准化
    X = StandardScaler().fit_transform(X)
    # 4，拆分测试、验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
    # 5，训练
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    print("Linear Regression:", lin_model.score(X_test, y_test).round(3))

    # 6，预测
    lin_pred = lin_model.predict(X_test)
    ser_lin_pred = pd.Series(np.round(lin_pred, 1))
    ser_lin_pred.name = "Predicted"
    ser_y_test = pd.Series(y_test)
    ser_y_test = ser_y_test.reset_index(drop=True)
    ser_y_test.name = "Actual"

    difference = ser_y_test - ser_lin_pred
    difference.name = 'Actual - Prediction'

    prediction = pd.concat([ser_y_test, ser_lin_pred, difference], axis=1).reset_index(drop=True)
    print(prediction)

    # singleXDemo()



