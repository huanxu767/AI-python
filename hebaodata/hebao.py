from numpy import log
from pandas import DataFrame as df
import pandas as pd
import pymysql

def getData(sql,cols):
    conn = pymysql.connect(host='127.0.0.1'  # 连接名称，默认127.0.0.1
                           , user='root'  # 用户名
                           , passwd='xuhuan'  # 密码
                           , port=3306  # 端口，默认为3306
                           , db='xdb'  # 数据库名称
                           , charset='utf8'  # 字符编码
                           )
    cur = conn.cursor()  # 生成游标对象
    cur.execute(sql)  # 执行SQL语句
    data = cur.fetchall()  # 通过fetchall方法获得数据
    cur.close()  # 关闭游标
    conn.close()  # 关闭连接
    return pd.DataFrame(data,columns=cols)

 # 皮尔逊相关系数
def pearsonCorrelDemo():
    # A = [1, 2, 3, 4, 5, 6]
    # B = [1, 2, 3, 4, 5, 6]
    # A1 = pd.Series(A)
    # B1 = pd.Series(B)
    # corr = B1.corr(A1, method='pearson')
    # print(corr)
    # dataframe计算#
    df = pd.DataFrame({'a':[1, 3, 6, 9, 0, 3],'b':[3, 5, 1, 4, 11, 3]})
    print(df)
    corr = df.corr(method='pearson')
    print(corr)


def orderCorr(df):
    _5GDf = df.query("product == '5G消费券'")[['num']]
    _shsxDf = df.query("product == '实时授信(储蓄卡)'")[['num']]
    _zfbDf = df.query("product == '支付宝预授权'")[['num']]
    _shsxDf.reset_index(inplace=True, drop=True)
    _zfbDf.reset_index(inplace=True, drop=True)
    df_destiny = pd.concat([_5GDf, _shsxDf, _zfbDf], axis=1, keys=['5G消费券', '实时授信', '支付宝'])
    corr = df_destiny.corr(method='pearson')
    print(corr)


def rateCorr(df):
    _5GDf = df.query("product == '5G消费券'")[['rate']]
    _shsxDf = df.query("product == '实时授信(储蓄卡)'")[['rate']]
    _zfbDf = df.query("product == '支付宝预授权'")[['rate']]
    _shsxDf.reset_index(inplace=True, drop=True)
    _zfbDf.reset_index(inplace=True, drop=True)
    df_destiny = pd.concat([_5GDf, _shsxDf, _zfbDf], axis=1, keys=['5G消费券', '实时授信', '支付宝'])

    # print(df_destiny)
    corr = df_destiny.corr(method='pearson')
    print(corr)

if __name__ == '__main__':

    sql = '''
        select a.`业务模式` product, `month`,count(1) num
        from hebao_order_fact a group by a.`业务模式`, `month`
        '''
    columns = ['product','month','num']
    df = getData(sql,columns)
    df['rate'] = df.groupby('month')['num'].transform(lambda x: x/ x.sum())
    orderCorr(df)
    rateCorr(df)
    # df_mask = df[df['product'] == '5G消费券']
    # print(df_mask)


    # 占比相关性
    # print(_5GDf['num'].sum())


    _