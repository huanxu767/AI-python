#运输问题求解：使用Vogel逼近法寻找初始基本可行解
import numpy as np
import copy
import pandas as pd

def TP_split_matrix(mat):
    c=mat[:-1,:-1]
    a=mat[:-1,-1]
    b=mat[-1,:-1]
    return (c,a,b)

def TP_vogel(var): #Vogel法代码，变量var可以是以numpy.ndarray保存的运输表，或以tuple或list保存的(成本矩阵,供给向量,需求向量)
    import numpy
    typevar1=type(var)==numpy.ndarray
    typevar2=type(var)==tuple
    typevar3=type(var)==list
    if typevar1==False and typevar2==False and typevar3==False:
        print('>>>非法变量<<<')
        (cost,x)=(None,None)
    else:
        if typevar1==True:
            [c,a,b]=TP_split_matrix(var)
        elif typevar2==True or typevar3==True:
            [c,a,b]=var
        cost=copy.deepcopy(c)
        x=np.zeros(c.shape)
        M=pow(10,9)
        for factor in c.reshape(1,-1)[0]:
            while int(factor)!=M:
                if np.all(c==M):
                    break
                else:
                    print('c:\n',c)
                    #获取行/列最小值数组
                    row_mini1=[]
                    row_mini2=[]
                    for row in range(c.shape[0]):
                        Row=list(c[row,:])
                        row_min=min(Row)
                        row_mini1.append(row_min)
                        Row.remove(row_min)
                        row_2nd_min=min(Row)
                        row_mini2.append(row_2nd_min)
                    #print(row_mini1,'\n',row_mini2)
                    r_pun=[row_mini2[i]-row_mini1[i] for i in range(len(row_mini1))]
                    print('行罚数：',r_pun)
                    #计算列罚数
                    col_mini1=[]
                    col_mini2=[]
                    for col in range(c.shape[1]):
                        Col=list(c[:,col])
                        col_min=min(Col)
                        col_mini1.append(col_min)
                        Col.remove(col_min)
                        col_2nd_min=min(Col)
                        col_mini2.append(col_2nd_min)
                    c_pun=[col_mini2[i]-col_mini1[i] for i in range(len(col_mini1))]
                    print('列罚数：',c_pun)
                    pun=copy.deepcopy(r_pun)
                    pun.extend(c_pun)
                    print('罚数向量：',pun)
                    max_pun=max(pun)
                    max_pun_index=pun.index(max(pun))
                    max_pun_num=max_pun_index+1
                    print('最大罚数：',max_pun,'元素序号：',max_pun_num)
                    if max_pun_num<=len(r_pun):
                        row_num=max_pun_num
                        print('对第',row_num,'行进行操作：')
                        row_index=row_num-1
                        catch_row=c[row_index,:]
                        print(catch_row)
                        min_cost_colindex=int(np.argwhere(catch_row==min(catch_row)))
                        print('最小成本所在列索引：',min_cost_colindex)
                        if a[row_index]<=b[min_cost_colindex]:
                            x[row_index,min_cost_colindex]=a[row_index]
                            c1=copy.deepcopy(c)
                            c1[row_index,:]=[M]*c1.shape[1]
                            b[min_cost_colindex]-=a[row_index]
                            a[row_index]-=a[row_index]
                        else:
                            x[row_index,min_cost_colindex]=b[min_cost_colindex]
                            c1=copy.deepcopy(c)
                            c1[:,min_cost_colindex]=[M]*c1.shape[0]
                            a[row_index]-=b[min_cost_colindex]
                            b[min_cost_colindex]-=b[min_cost_colindex]
                    else:
                        col_num=max_pun_num-len(r_pun)
                        col_index=col_num-1
                        print('对第',col_num,'列进行操作：')
                        catch_col=c[:,col_index]
                        print(catch_col)
                        #寻找最大罚数所在行/列的最小成本系数
                        min_cost_rowindex=int(np.argwhere(catch_col==min(catch_col)))
                        print('最小成本所在行索引：',min_cost_rowindex)
                        #计算将该位置应填入x矩阵的数值（a,b中较小值）
                        if a[min_cost_rowindex]<=b[col_index]:
                            x[min_cost_rowindex,col_index]=a[min_cost_rowindex]
                            c1=copy.deepcopy(c)
                            c1[min_cost_rowindex,:]=[M]*c1.shape[1]
                            b[col_index]-=a[min_cost_rowindex]
                            a[min_cost_rowindex]-=a[min_cost_rowindex]
                        else:
                            x[min_cost_rowindex,col_index]=b[col_index]
                            #填入后删除已满足/耗尽资源系数的行/列，得到剩余的成本矩阵，并改写资源系数
                            c1=copy.deepcopy(c)
                            c1[:,col_index]=[M]*c1.shape[0]
                            a[min_cost_rowindex]-=b[col_index]
                            b[col_index]-=b[col_index]
                    c=c1
                    print('本次迭代后的x矩阵：\n',x)
                    print('a:',a)
                    print('b:',b)
                    print('c:\n',c)
                if np.all(c==M):
                    print('【迭代完成】')
                    print('-'*60)
                else:
                    print('【迭代未完成】')
                    print('-'*60)
        total_cost=np.sum(np.multiply(x,cost))
        if np.all(a==0):
            if np.all(b==0):
                print('>>>供求平衡<<<')
            else:
                print('>>>供不应求，需求方有余量<<<')
        elif np.all(b==0):
            print('>>>供大于求，供给方有余量<<<')
        else:
            print('>>>无法找到初始基可行解<<<')
        print('>>>初始基本可行解x*：\n',x)
        print('>>>当前总成本：',total_cost)
        [m,n]=x.shape
        varnum=np.array(np.nonzero(x)).shape[1]
        if varnum!=m+n-1:
            print('【注意：问题含有退化解】')
    return (cost,x)

path=r'C:\Users\spurs\Desktop\MCM_ICM\Data files\TP_PPT_Sample1.xlsx'
mat=pd.read_excel(path,header=None).values
#c=np.array([[3,11,3,10],[1,9,2,8],[7,4,10,5]])
#a=np.array([7,4,9])
#b=np.array([3,6,5,6])
[c,x]=TP_vogel(mat)
#[c,x]=TP_vogel([c,a,b])
