import numpy as np
import pandas as pd

# https://pandas.pydata.org/docs/pandas.pdf
# https://www.runoob.com/pandas/pandas-tutorial.html
# https://pandas.pydata.org/docs/user_guide/10min.html


def basic():

    df = pd.DataFrame({'points': [25, 12, 15, 14],
                       'assists': [5, 7, 13, 12]})

    print(df)


if __name__ == '__main__':
    basic()
