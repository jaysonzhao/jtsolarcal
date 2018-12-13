#!/usr/bin/env python3

import connexion
import numpy as np
import pandas as pd
from connexion import NoContent

from dfa import dfa


# 计算平均值
def post_mean(array: list):
    try:
        # 求均值
        calresult = np.mean(array)
        return '{result}'.format(result=calresult)
    except:
        return NoContent, 404


# 计算变化率标准差
def post_rocsd(array: list):
    try:
        # 计算变化率
        result = []
        it = iter(array)
        n1 = float(it.__next__())
        for n2 in it:
            result.append((n2 - n1) / n1)
            n1 = float(n2)
        # 求变化率标准差
        calresult = np.std(result, ddof=1)
        return '{result}'.format(result=calresult)
    except:
        return NoContent, 404


# 计算DFA 去趋势波动分析指数
def post_dfa(array: list):
    try:
        scales, fluct, alpha = dfa(array)
        calresult = alpha
        return '{result}'.format(result=calresult)
    except:
        return NoContent, 404


# 计算方差
def post_var(array: list):
    try:
        calresult = np.var(array)
        return '{result}'.format(result=calresult)
    except:
        return NoContent, 404


# 计算滚动方差最大值
def post_rollingvarmax(array: list, window: int):
    try:
        calresult = pd.Series(array).rolling(window=window, center=False).var(ddof=1)
        return '{result}'.format(result=np.max(calresult))
    except:
        return NoContent, 404

# 计算滚动方差平均值
def post_rollingvarmean(array: list, window: int):
    try:
        calresult = pd.Series(array).rolling(window=window, center=False).var(ddof=1)
        return '{result}'.format(result=np.mean(calresult))
    except:
        return NoContent, 404

if __name__ == '__main__':
    app = connexion.FlaskApp(__name__, port=9090, specification_dir='swagger/')
    app.add_api('solarcalhub-api.yaml', arguments={'title': 'Solar Calculation Hub'})
    app.run()
