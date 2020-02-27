import glob
import os
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 1000)  # 设定打印的限制
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

path = os.getcwd()
file = glob.glob(os.path.join(path, "*.csv"))
dl = []
for f in file:
    dl.append(pd.read_csv(f))
df = pd.concat(dl)
# df = df.fillna(0)

# 删除错误值
# df_cut = df[df['氨氮'] < 20]
# df_cut = df_cut[df_cut['pH'] > 0]

# df_back = df_cut

df_cut = df[['time','deviceid' ,'pH', '氨氮', '浊度', '叶绿素', '电导率', '水温']]
print(df_cut)
df_cut = df_cut.dropna(axis=0, how='any')
group1 = df_cut.groupby('deviceid').size()
group1.plot('bar')
plt.show()
# print(np.where(~np.isnan(df_cut)))

# df_cut = df_cut.dropna(axis=0, how='any')

# df_cut = df_cut.iloc[0:1000]
print(df_cut.describe())
# print(df_cut[df_cut['浊度'] > 80])
# df_time = df_cut[['time']]
# # print(df_time)
# df_cut_time = df_cut[['pH', 'DO', 'COD', '氨氮', 'ORP', '浊度', '叶绿素', '电导率', '水温']]
# df_cut_time = (df_cut_time-df_cut_time.min())/(df_cut_time.max()-df_cut_time.min())
# df_clean = pd.concat([df_time, df_cut_time], axis=1)
# print(df_clean.corr())
# sns.heatmap(df_clean.corr())
#
# # df_cut.plot(y=['DO', '浊度'])
#
# # df_back.plot(x='ORP', y=['pH'])
# plt.show()