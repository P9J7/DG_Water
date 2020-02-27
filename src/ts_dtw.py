import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tslearn.utils import to_time_series
from tslearn.utils import to_time_series_dataset
from tslearn.datasets import UCR_UEA_datasets
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler
from tslearn.clustering import GlobalAlignmentKernelKMeans
from tslearn.clustering import silhouette_score as ss
from tslearn.clustering import KShape
from tslearn.metrics import sigma_gak
from tslearn.metrics import cdist_dtw

pd.set_option('display.max_columns', 1000)  # 设定打印的限制
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False



path = os.getcwd()
file = glob.glob(os.path.join(path, "DY桥头-石马河汇入（动态巡查23-B）.csv"))
# file = glob.glob(os.path.join(path, "DY大朗-松木山水大陂海(动态巡查B04).csv"))
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
data = data.groupby(data.index.day)
ts_list = list()
for f in data:
    # print(f[1]['氨氮'])
    ts_list.append(list(f[1]['氨氮']))

# formatted_time_series = to_time_series(df['氨氮'])
# formatted_dataset = to_time_series_dataset([df['time'], df['氨氮']])
# print(formatted_time_series.shape)
# print(formatted_dataset.shape)
seed = 0
np.random.seed(seed)
my_first_time_series = [1, 3, 4, 2]
my_second_time_series = [1, 2, 3, 4]
my_third_time_series = [4, 3, 2, 1]
my_forth_time_series = [2, 6, 8, 9, 20]
# formatted_dataset = to_time_series_dataset([my_first_time_series, my_second_time_series, my_third_time_series, my_forth_time_series])
formatted_dataset = to_time_series_dataset(ts_list)
X_train = formatted_dataset
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = TimeSeriesResampler(sz=80).fit_transform(X_train)
sz = X_train.shape[1]
best_score = 0.0
best_n_cluster = None
best_y_pred = None
best_cluster_centers_ = None
for i in np.arange(2, 7):
    sdtw_km = TimeSeriesKMeans(n_clusters=i,
                           metric="softdtw",
                           metric_params={"gamma": .01},
                           verbose=True,
                           random_state=seed)
    y_pred = sdtw_km.fit_predict(X_train)
    score = ss(X_train, sdtw_km.labels_, metric='softdtw')
    if score > best_score:
        best_score = score
        best_n_cluster = i
        best_y_pred = y_pred
        best_cluster_centers_ = sdtw_km.cluster_centers_
# ks = KShape(n_clusters=5, verbose=True, random_state=seed)
# y_pred = ks.fit_predict(X_train)
# gak_km = GlobalAlignmentKernelKMeans(n_clusters=5,
#                                      sigma=sigma_gak(X_train),
#                                      n_init=20,
#                                      verbose=True,
#                                      random_state=seed)
# y_pred = gak_km.fit_predict(X_train)
print("聚%d"%best_n_cluster + "类的平均轮廓系数最高为：%f"%best_score)
plt.figure(figsize=(20,10))
# print(sdtw_km.labels_)
for yi in range(best_n_cluster):
    plt.subplot(best_n_cluster, 1,  1 + yi)
    for xx in X_train[best_y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(best_cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    # plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Soft-DTW $k$-means")

plt.tight_layout()
plt.show()
