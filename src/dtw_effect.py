import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 1000)  # 设定打印的限制
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False
# file = glob.glob(os.path.join(path, "DY桥头-石马河汇入（动态巡查23-B）.csv"))

def get_common_seq(best_path, threshold=1):
    com_ls = []
    pre = best_path[0]
    length = 1
    for i, element in enumerate(best_path):
        if i == 0:
            continue
        cur = best_path[i]
        if cur[0] == pre[0] + 1 and cur[1] == pre[1] + 1:
            length = length + 1
        else:
            com_ls.append(length)
            length = 1
        pre = cur
    com_ls.append(length)
    return list(filter(lambda num: True if threshold < num else False, com_ls))


def calculate_attenuate_weight(seqLen, com_ls):
    weight = 0
    for comlen in com_ls:
        weight = weight + (comlen * comlen) / (seqLen * seqLen)
    return 1 - math.sqrt(weight)


def best_path(paths):
    """Compute the optimal path from the nxm warping paths matrix."""
    i, j = int(paths.shape[0] - 1), int(paths.shape[1] - 1)
    p = []
    if paths[i, j] != -1:
        p.append((i - 1, j - 1))
    while i > 0 and j > 0:
        c = np.argmin([paths[i - 1, j - 1], paths[i - 1, j], paths[i, j - 1]])
        if c == 0:
            i, j = i - 1, j - 1
        elif c == 1:
            i = i - 1
        elif c == 2:
            j = j - 1
        if paths[i, j] != -1:
            p.append((i - 1, j - 1))
    p.pop()
    p.reverse()
    return p


def TimeSeriesSimilarity(s1, s2):
    l1 = len(s1)
    l2 = len(s2)
    paths = np.full((l1 + 1, l2 + 1), np.inf)  # 全部赋予无穷大
    paths[0, 0] = 0
    for i in range(l1):
        for j in range(l2):
            d = s1[i] - s2[j]
            cost = d ** 2
            paths[i + 1, j + 1] = cost + min(paths[i, j + 1], paths[i + 1, j], paths[i, j])

    paths = np.sqrt(paths)
    s = paths[l1, l2]
    return s, paths.T

def ETL(ts1):
    df = pd.read_csv(ts1)
    df = df[['pH', '氨氮', '浊度', '叶绿素', '电导率', '水温']]
    df = df.dropna(axis=0, how='any')
    df = df[df['氨氮'] > 1]
    df = df['氨氮'].values
    return df

def mod_dtw(ts1, ts2):
    distance12, paths12 = TimeSeriesSimilarity(ts1, ts2)
    best_path12 = best_path(paths12)
    com_ls1 = get_common_seq(best_path12)
    weight12 = calculate_attenuate_weight(len(best_path12), com_ls1)
    return distance12*weight12

if __name__ == '__main__':
    df1 = ETL('DY桥头-石马河汇入（动态巡查23-B）.csv')
    df2 = ETL('DY干流-企石鸿发桥（动态巡查B40）.csv')
    df3 = ETL('DY大朗-松木山水大陂海(动态巡查B04).csv')

    print(mod_dtw(df1, df2))
    print(mod_dtw(df1, df3))
    # # 测试数据
    # s1 = np.array([1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1])
    # s2 = np.array([0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2])
    # s3 = np.array([0.8, 1.5, 0, 1.2, 0, 0, 0.6, 1, 1.2, 0, 0, 1, 0.2, 2.4, 0.5, 0.4])
    #
    # # 原始算法
    # distance12, paths12 = TimeSeriesSimilarity(df1, df2)
    # distance13, paths13 = TimeSeriesSimilarity(df1, df3)
    #
    # # print("更新前s1和s2距离：" + str(distance12))
    # # print("更新前s1和s3距离：" + str(distance13))
    #
    # best_path12 = best_path(paths12)
    # best_path13 = best_path(paths13)
    #
    # # 衰减系数
    # com_ls1 = get_common_seq(best_path12)
    # com_ls2 = get_common_seq(best_path13)
    #
    # # print(len(best_path12), com_ls1)
    # # print(len(best_path13), com_ls2)
    # weight12 = calculate_attenuate_weight(len(best_path12), com_ls1)
    # weight13 = calculate_attenuate_weight(len(best_path13), com_ls2)
    #
    # # 更新距离
    # print("更新后s1和s2距离：" + str(distance12 * weight12))
    # print("更新后s1和s3距离：" + str(distance13 * weight13))