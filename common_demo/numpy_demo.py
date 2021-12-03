# https://numpy.org/doc/stable/

import numpy as np
from numpy import pi

def basic():
    a = np.arange(15).reshape(3,5)
    print(a)
    # np.arange(0, 2, 0.3)

    # b = np.array([2, 3, 4])
    # print(b)
    #
    # c = np.array([(1.5, 2, 3), (4, 5, 6)])
    # print(c)

    # np.zeros((3, 4))
    #
    # np.ones((2, 3, 4))
    #
    # np.linspace(0, 2, 9)
    #
    # x = np.linspace(0, 2 * pi, 100)


    # 与许多矩阵语言不同，乘积运算符*在 NumPy 数组中按元素进行运算。可以使用@运算符（在python>=3.5中）或dot函数或方法来执行矩阵乘积

    A = np.array([[1, 1],[0, 1]])
    B = np.array([[2, 0],[3, 4]])

    print(A*B) # elementwise product
    print(A.dot(B))# matrix product
    print(A @ B) # another matrix product

def count():
    a = np.arange(12).reshape(3,4)
    print(a.sum(axis=0)) # sum of each column
    print(a.min(axis=1)) # sum of each column
    print(a.cumsum(axis=1))# cumulative sum along each row


if __name__ == '__main__':
    # basic()
    count()