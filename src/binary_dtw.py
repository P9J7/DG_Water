import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import src.dtw_effect as de
import src.dtw_match as dm
import src.dis_dtw as dd

pd.set_option('display.max_columns', 1000)  # 设定打印的限制
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False


def diff_pre(df, start, end):
    df = df[start:end]
    df = df.resample('2H').mean()
    df = df.diff()
    df.dropna(inplace=True)
    # df['氨氮'] = df['氨氮'].apply(lambda x: 1 if x > 0 else 0)
    return df


def binary_pre(df, start, end):
    df = df[start:end]
    df = df.resample('2H').mean()
    df = df.diff()
    df.dropna(inplace=True)
    df['氨氮'] = df['氨氮'].apply(lambda x: 1 if x > 0 else 0)
    return df


if __name__ == '__main__':
    df1 = de.ETL('data/DY寮步-横竹河虚舟路（动态巡查B18）.csv')
    df2 = de.ETL('data/DY寮步-黄沙河下（动态巡查B17）.csv')
    # df3 = de.ETL('data/DY东城-筷子河（动态巡查B36）.csv')
    start = '2020-02-11'
    end = '2020-03-01'
    # df1 = diff_pre(df1, start, end)
    df1 = binary_pre(df1, start, end)
    # df2 = binary_pre(df2, start, end)
    # df2 = diff_pre(df2, start, end)
    df2 = binary_pre(df2, start, end)
    # df1.plot()
    # df2.plot()
    # df2.plot()
    plt.show()
    length = 6
    i = 0
    for start in range(24, 240, 24):
        match_dtw = dm.match_length(df1, df2, start, length)
        pattern_dtw = df1[start:start + length]
        plt.subplot(3, 3, i + 1)
        plt.plot(pattern_dtw['氨氮'].values)
        plt.plot(match_dtw['氨氮'].values, linewidth=3)
        i = i + 1
    plt.show()

