import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import src.dtw_effect as de
from tslearn.metrics import dtw, dtw_path
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

pd.set_option('display.max_columns', 1000)  # 设定打印的限制
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False


def match_length(df1, df2, start, length):
    df1 = df1[start:start+length]
    time_start = df1.index[0]
    print("匹配的时间起点是{}".format(time_start))
    # print("相应的数据序列为{}".format(df1))
    window = 24
    match_time_start = None
    for j in df2.index:
        if j > time_start:
            print("下游匹配到的时间起点是{}".format(j))
            match_time_start = df2.index.tolist().index(j)
            print("下游对应的序号为{}".format(match_time_start))
            break
    dtw_score = list()
    match_df2 = None
    # match_df2 = df2[347:347 + length]
    # 改动了窗口的起始点
    for o in range(3, window):
        cut_df2 = df2[match_time_start + o:match_time_start + length + o]
        mod_dtw, _ = de.TimeSeriesSimilarity(df1['氨氮'].values, cut_df2['氨氮'].values)
        # mod_dtw, _ = de.TimeSeriesSimilarity(df1.tolist(), cut_df2.tolist())
        dtw_score.append(mod_dtw)
        if dtw_score.index(max(dtw_score)) == o:
            print("最大的dtw得分偏移了{}".format(o))
            print("相应的开始时间点为{}".format(cut_df2.index[0]))
            match_df2 = cut_df2
    return match_df2


def ts_match(df1, df2, start, length):
    df1 = df1[start:start+length]
    time_start = df1.index[0]
    print("匹配的时间起点是{}".format(time_start))
    # print("相应的数据序列为{}".format(df1))
    window = 48
    match_time_start = None
    for j in df2.index:
        if j > time_start:
            print("下游匹配到的时间起点是{}".format(j))
            match_time_start = df2.index.tolist().index(j)
            print("下游对应的序号为{}".format(match_time_start))
            break
    dtw_score = list()
    match_df2 = None
    # match_df2 = df2[347:347 + length]
    for o in range(window):
        cut_df2 = df2[match_time_start + o:match_time_start + length + o]
        dtw_sim = dtw(df1['氨氮'].values, cut_df2['氨氮'].values, global_constraint="sakoe_chiba", sakoe_chiba_radius=3)
        # dtw_sim = dtw(TimeSeriesScalerMeanVariance().fit_transform(df1['氨氮'].values),
        #               TimeSeriesScalerMeanVariance().fit_transform(cut_df2['氨氮'].values),
        #               global_constraint="sakoe_chiba", sakoe_chiba_radius=3)
        # mod_dtw, _ = de.TimeSeriesSimilarity(df1.tolist(), cut_df2.tolist())
        dtw_score.append(dtw_sim)
        if dtw_score.index(max(dtw_score)) == o:
            print("最大的dtw得分偏移了{}".format(o))
            print("相应的开始时间点为{}".format(cut_df2.index[0]))
            match_df2 = cut_df2
    print("滑动窗口的得分列表为{}".format(dtw_score))
    return match_df2, dtw_score.index(max(dtw_score))


def pre_bin(df, time_start, time_end):
    df = df[time_start:time_end]
    df = df.resample('H').mean()
    # df['氨氮'] = np.squeeze(TimeSeriesScalerMeanVariance().fit_transform(df['氨氮'].values))
    # df = df.diff()
    # df.dropna(inplace=True)
    # bins = split_list(df, n)
    # df_dis = pd.cut(df['氨氮'], bins, labels=range(n))
    # df_dis = pd.qcut(df['氨氮'], n, labels=range(n))
    return df


if __name__ == '__main__':
    df1 = de.ETL('data/DY干流-茶山横岗埔(动态巡查B62).csv')
    df2 = de.ETL('data/DY干流-东城金桥水产市场（动态巡查B88）.csv')
    start = '2020-02-28'
    end = '2020-03-05'
    df1 = pre_bin(df1, start, end)
    # print(df1)
    df2 = pre_bin(df2, start, end)
    # print(df2)
    length = 24
    i = 0
    for start in range(0, 96, 24):
        match_dtw, idx = ts_match(df1, df2, start, length)
        pattern_dtw = df1[start:start+length]
        plt.subplot(2, 2, i+1)
        plt.plot(pattern_dtw['氨氮'].values)
        plt.plot(match_dtw['氨氮'].values, linewidth=3)
        plt.title("最匹配的曲线偏移了{}单位".format(idx))
        i = i + 1
    plt.show()
