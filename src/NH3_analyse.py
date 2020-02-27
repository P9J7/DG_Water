import glob
import os
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


pd.set_option('display.max_columns', 1000)  # 设定打印的限制
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False



path = os.getcwd()
file = glob.glob(os.path.join(path, "DY桥头-石马河汇入（动态巡查23-B）.csv"))
dl = []
for f in file:
    dl.append(pd.read_csv(f))
df = pd.concat(dl)
df = df.dropna(axis=0, how='any')
df = df[df['氨氮'] > 1]
df = df[['time', '氨氮']]
df['time'] = pd.to_datetime(df['time'])
data = df.loc[:, ['氨氮']]
data = data.set_index(df.time)
weekly = data.resample('W').sum()
weekly.plot(style=[':', '--', '-'])
daily = data.resample('D').sum()
daily.plot(style=[':', '--', '-'])
by_time = data.groupby(data.index.time).mean()
hourly_ticks = 4*60*60*np.arange(6)
by_time.plot(xticks=hourly_ticks, style=[':', '--', '-'])
plt.show()
# data.plot()
# plt.show()
# date_rng = pd.date_range(start='20/12/2019', end='20/01/2020')
# print(date_rng)
# for i in date_rng:
#     for j in df.time:
#         if j in i:
#             print(j)

# print(df.describe())
# print(df)
# df = df.iloc[0:100]
# df.plot(x='time', y=['氨氮'])
# plt.show()
