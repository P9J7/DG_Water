import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import src.dtw_effect as de
import src.dtw_match as dm

pd.set_option('display.max_columns', 1000)  # 设定打印的限制
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False


def split_list(df, n):
    q3 = float(df.quantile(0.75))
    q1 = float(df.quantile(0.25))
    iqr = q3 - q1
    min_line = q1 - 1.5 * iqr
    max_line = q3 + 1.5 * iqr
    list_split = None
    if min_line > 0:
        step = (max_line - min_line) / (n - 1)
        list_split = [(min_line + x * step) for x in range(n - 1)]
        list_split.insert(0, 0)
    else:
        step = (max_line - 0) / n
        list_split = [(0 + x * step) for x in range(n)]
    list_split.append(float(df.max()))
    return list_split


def pre_bin(df, time_start, time_end, n):
    df = df[time_start:time_end]
    df = df.resample('2H').mean()
    # bins = split_list(df, n)
    # df_dis = pd.cut(df['氨氮'], bins, labels=range(n))
    df_dis = pd.qcut(df['氨氮'], n, labels=range(n))
    return df_dis


if __name__ == '__main__':
    df1 = de.ETL('data/DY寮步-黄沙河下（动态巡查B17）.csv')
    df2 = de.ETL('data/DY东城-筷子河（动态巡查B36）.csv')
    k = 10
    start = '2020-02-11'
    end = '2020-03-05'
    df1 = pre_bin(df1, start, end, k)
    print(pd.value_counts(df1))
    df2 = pre_bin(df2, start, end, k)
    print(pd.value_counts(df2))
    plt.plot(df1.tolist())
    plt.show()
    # print(df1)
    # print(df2)
    length = 6
    i = 0
    for start in range(24, 240, 24):
        match_dtw = dm.match_length(df1, df2, start, length)
        pattern_dtw = df1[start:start + length]
        plt.subplot(3, 3, i + 1)
        plt.plot(pattern_dtw.tolist())
        plt.plot(match_dtw.tolist(), linewidth=3)
        i = i + 1
    plt.show()
