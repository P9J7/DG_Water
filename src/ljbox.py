
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns     #seaborn画出的图更好看，且代码更简单，缺点是可塑性差
from statsmodels.graphics.tsaplots import plot_acf  #自相关图
from statsmodels.tsa.stattools import adfuller as ADF  #平稳性检测
from statsmodels.graphics.tsaplots import plot_pacf    #偏自相关图
from statsmodels.stats.diagnostic import acorr_ljungbox    #白噪声检验
from statsmodels.tsa.arima_model import ARIMA
#seaborn 是建立在matplotlib之上的


#jupyter中文显示是方框，加入下面两行即可显示中文，若嫌麻烦，也可去网上搜索如何永久显示中文
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# pylab.rcParams['figure.figsize'] = (10, 6)   #设置输出图片大小
sns.set(color_codes=True) #seaborn设置背景

#读取数据，指定日期列为指标，Pandas自动将“日期”列识别为Datetime格式
df = pd.read_csv('DY桥头-石马河汇入（动态巡查23-B）.csv')
df = df[['pH', '氨氮', '浊度', '叶绿素', '电导率', '水温']]
df = df.dropna(axis=0, how='any')
df = df[df['氨氮'] > 1]
df = df['氨氮']
# df = df.diff()
print(df)
#时序图
plot_acf(df).show()
plot_pacf(df).show()
plt.show()