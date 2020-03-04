import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.dtw_effect as de
import statsmodels.api as sm
from pandas import concat
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
pd.set_option('display.max_columns', 1000)  # 设定打印的限制
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False


def plot_acf(data):
    sm.graphics.tsa.plot_acf(data, lags=50)
    plt.figure(figsize=(12, 6))
    plt.show()


def plot_pacf(data):
    sm.graphics.tsa.plot_pacf(data, lags=50)
    plt.figure(figsize=(12, 6))
    plt.show()


def lag_corr(data):
    dataframe = concat([data.shift(1), data], axis=1)
    dataframe.columns = ['t-1', 't+1']
    result = dataframe.corr()
    print(result)


def plot_acf_pacf(data):
    data.plot()
    plt.show()
    plot_acf(data)
    plot_pacf(data)


def diff_plot_acf_pacf(data, n):
    data = data.diff(n)
    data.dropna(inplace=True)
    data.plot()
    plt.show()
    plot_acf(data)
    plot_pacf(data)


def ar_model(data):
    # split dataset
    X = data.values
    train, test = X[1:len(X) - 7], X[len(X) - 7:]
    # train autoregression
    model = AR(train)
    model_fit = model.fit()
    print('Lag: %s' % model_fit .k_ar)
    print('Coefficients: %s' % model_fit.params)
    # make predictions
    predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
    for i in range(len(predictions)):
        print('predicted=%f, expected=%f' % (predictions[i], test[i]))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot results
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()


if __name__ == '__main__':
    df1 = de.ETL('DY寮步-黄沙河下（动态巡查B17）.csv')
    # df1 = de.ETL('DY桥头-石马河汇入（动态巡查23-B）.csv')
    ar_model(df1)
    # diff_plot_acf_pacf(df1, 2)
    # lag_corr(df1)