import pandas as pd
import numpy as np
# time_stamp = pd.date_range(start='2018-1-1', end='2019-12-31', freq='H')
# data = pd.DataFrame(time_stamp)
# data.to_csv("time_stamp.csv", index=False, header=0)
d1= ['铜版纸', 0,'【哑膜】']
d2= ['铜版纸',50,'【5】']
d3= ['铜版纸',300,'【哑膜】']
d4= ['铜版纸',300,'【1】']
data = pd.DataFrame(data=[d1, d2, d3,d4,],columns=['纸张', '克重','覆膜'])
data['克重'] = data['克重'].replace(0, np.nan)
print(data)