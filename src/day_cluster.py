import glob
import os
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn import metrics



pd.set_option('display.max_columns', 1000)  # 设定打印的限制
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False



path = os.getcwd()
# file = glob.glob(os.path.join(path, "DY桥头-石马河汇入（动态巡查23-B）.csv"))
file = glob.glob(os.path.join(path, "DY大朗-松木山水大陂海(动态巡查B04).csv"))
dl = []
for f in file:
    dl.append(pd.read_csv(f))
df = pd.concat(dl)
df = df[['time', '氨氮']]
df = df.dropna(axis=0, how='any')
df = df[df['氨氮'] > 1]
df['time'] = pd.to_datetime(df['time'])
data = df.loc[:, ['氨氮']]
data = data.set_index(df.time)

by_time = data
by_time['时间'] = by_time.index
by_time['秒'] = by_time['时间'].dt.hour*3600 + by_time['时间'].dt.minute*60 + by_time['时间'].dt.second
by_time['小时'] = by_time['时间'].dt.time
by_time = by_time.set_index(by_time['小时'])
by_time = by_time.drop(columns=['时间', '小时'])
print(by_time)
# print(by_time.index)
hourly_ticks = 2*60*60*np.arange(12)

norm_second = by_time["秒"].apply(lambda x: (x - by_time["秒"].min()) / (by_time["秒"].max() - by_time["秒"].min()))
print(by_time['秒'].max(), by_time['氨氮'].max())
by_time = by_time.drop("秒", axis=1)
by_time["秒"] = norm_second
norm_nh3 = by_time["氨氮"].apply(lambda x: (x - by_time["氨氮"].min()) / (by_time["氨氮"].max() - by_time["氨氮"].min()))
by_time = by_time.drop("氨氮", axis=1)
by_time["氨氮"] = norm_nh3


by_time.plot(kind='scatter', x='秒', y='氨氮')
# kmeans = KMeans(n_clusters=8)
# kmeans.fit(by_time)
# score = metrics.calinski_harabaz_score(by_time, kmeans.labels_)
# print(score)
dbscans = DBSCAN(eps=0.03, min_samples=5)
dbscans.fit(by_time)
plt.figure(figsize=(6.4,4.8))
colors = ['b', 'g', 'r', 'y', 'c', 'k', 'm', '#FF00FF', '#00FF00']
markers = ['o', 's', 'D', 'x', 'H', '^', '>', 'v', '-', '--']
for i,l in enumerate(dbscans.labels_):
     plt.plot(by_time['秒'][i]*86386, by_time['氨氮'][i]*14.25,color=colors[l],marker='x',ls='None')
plt.show()

# by_time.plot()
# date_rng = pd.date_range(start='20/12/2019', end='20/01/2020', freq='h')
# by_time = by_time.set_index(date_rng)
#
# hourly_ticks = 2*60*60*np.arange(13)
# by_time.plot(style=[':', '--', '-'], kind='scatter', x=date_rng, y=by_time['氨氮'])
plt.show()
