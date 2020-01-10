import glob
import os
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 1000)  # 设定打印的限制
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

path = os.getcwd()
file = glob.glob(os.path.join(path, "*.csv"))
print(file)
dl = []
for f in file:
    dl.append(pd.read_csv(f))
df = pd.concat(dl)
# print(df.loc[df['水质等级'] == 8902.4])
# df = df[~df['deviceid'].isin([928321083])]
# df.groupby(['deviceid']).浊度.mean().plot.bar()
# plt.tight_layout()
# plt.show()
print(df.columns)
df = df.fillna(0)
pic1 = df.groupby(['deviceid']).count()['devicename']
plt.show()
# plt.scatter(df['浊度'], df['水质等级'], 50, marker='+', linewidths=2, alpha=0.8, color='blue')
# plt.xlabel('浊度')
# plt.ylabel('水质等级')
# plt.xlim(0, 600)
# plt.ylim(0, 600)
# plt.grid(color='#95a5a6', linestyle='--', linewidth='1', axis='both', alpha=0.4)
# plt.show()
# data = np.array(df[['氨氮', 'COD']])
# clf = KMeans(n_clusters=7)
# clf = clf.fit(data)
# print(clf.cluster_centers_)
# df['label'] = clf.labels_
# # print(df.head())
# df0 = df.loc[df["label"] == 0]
# df1 = df.loc[df["label"] == 1]
# df2 = df.loc[df["label"] == 2]
# df3 = df.loc[df["label"] == 3]
# df4 = df.loc[df["label"] == 4]
# df5 = df.loc[df["label"] == 5]
# df6 = df.loc[df["label"] == 6]

# 绘制聚类结果的散点图
# plt.scatter(df0['氨氮'], df0['COD'], 50, color='#99CC01', marker='+', linewidth=2, alpha=0.8)
# plt.scatter(df1['氨氮'], df1['COD'], 50, color='#FE0000', marker='+', linewidth=2, alpha=0.8)
# plt.scatter(df2['氨氮'], df2['COD'], 50, color='#0000FE', marker='+', linewidth=2, alpha=0.8)
# plt.scatter(df3['叶绿素'], df3['水质等级'], 50, color='yellow', marker='+', linewidth=2, alpha=0.8)
# plt.scatter(df4['叶绿素'], df4['水质等级'], 50, color='black', marker='+', linewidth=2, alpha=0.8)
# plt.scatter(df5['叶绿素'], df5['水质等级'], 50, color='pink', marker='+', linewidth=2, alpha=0.8)
# plt.scatter(df6['叶绿素'], df6['水质等级'], 50, color='green', marker='+', linewidth=2, alpha=0.8)
# plt.xlabel('氨氮')
# plt.ylabel('COD')
# plt.xlim(0, 25)
# plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='both', alpha=0.4)
# plt.show()
