import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
pd.set_option('display.max_columns', 1000)  # 设定打印的限制
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False
# file = glob.glob(os.path.join(path, "DY桥头-石马河汇入（动态巡查23-B）.csv"))

df = pd.read_csv('DY大朗-松木山水大陂海(动态巡查B04).csv')
df = df[['pH', '氨氮', '浊度', '叶绿素', '电导率', '水温']]
df = df.dropna(axis=0, how='any')
df = df[df['氨氮'] > 1]
# spearman kendall
overall_pearson_r = df.corr()
print(f"Pandas computed Pearson r: \n{overall_pearson_r}")
# 输出：使用 Pandas 计算皮尔逊相关结果的 r 值：0.2058774513561943

r, p = stats.pearsonr(df.dropna()['氨氮'], df.dropna()['pH'])
print(f"Scipy computed Pearson r: {r} and p-value: {p}")
# 输出：使用 Scipy 计算皮尔逊相关结果的 r 值：0.20587745135619354，以及 p-value：3.7902989479463397e-51

# 计算滑动窗口同步性
f,ax=plt.subplots(figsize=(10,10))
df.rolling(window=30,center=True).median().plot(ax=ax)
ax.set(xlabel='Time',ylabel='Pearson r')
ax.set(title=f"Overall Pearson r")
plt.show()

# 设置窗口宽度，以计算滑动窗口同步性
r_window_size = 120
# 插入缺失值
df_interpolated = df.interpolate()
# 计算滑动窗口同步性
rolling_r = df_interpolated['氨氮'].rolling(window=r_window_size, center=True).corr(df_interpolated['叶绿素'])
f,ax=plt.subplots(2,1,figsize=(14,6),sharex=True)
df.rolling(window=30,center=True).median().plot(ax=ax[0])
ax[0].set(xlabel='Frame',ylabel='Smiling Evidence')
rolling_r.plot(ax=ax[1])
ax[1].set(xlabel='Frame',ylabel='Pearson r')
plt.suptitle("Smiling data and rolling window correlation")
plt.show()
