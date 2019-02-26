#!/usr/bin/env python3

import connexion
import numpy as np
import pandas as pd
import socket
from connexion import NoContent
from arch import arch_model  # GARCH(1,1)
from scipy import stats
from dfa import dfa
import scipy.signal as signal

# 计算平均值
def post_mean(array: list):
    try:
        # 求均值
        calresult = np.mean(array)
        return '{result}'.format(result=calresult)
    except:
        return NoContent, 404


# 计算标准差
def post_std(array: list):
    try:
        # 求标准差
        calresult = np.std(array)
        return '{result}'.format(result=calresult)
    except:
        return NoContent, 404

# 计算对数振荡率
def post_logVolatility(array: list):
    try:
        # 求对数振荡率
        logarr = np.log(array)
        calresult = np.std(logarr, ddof=1) / np.mean(logarr) / np.sqrt(1 / len(array))

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

# 计算最大波动率
def post_maxsqt(array: list):
    try:
        # 计算波动率 from GARCH(1,1)
        am = arch_model(array)
        res = am.fit()
        sqt_h = res.conditional_volatility
        calresult = np.max(sqt_h)
        return '{result}'.format(result=calresult)
    except:
        return NoContent, 404

# 计算极值点最大波动率
def post_maxpeaksqt(array: list):
    try:
        peakidx, _ = signal.find_peaks(array, prominence=1)
        if len(peakidx) == 0:
            return '{result}'.format(result=0)
        # 计算波动率 from GARCH(1,1)
        peakarr = np.array(array)
        amt = arch_model(peakarr[peakidx])
        rest = amt.fit()
        sqt_ht = rest.conditional_volatility
        calresult = np.max(sqt_ht)
        return '{result}'.format(result=calresult)
    except:
        return NoContent, 404

# 计算平均波动率
def post_meansqt(array: list):
    try:
        # 计算波动率 from GARCH(1,1)
        am = arch_model(array)
        res = am.fit()
        sqt_h = res.conditional_volatility
        calresult = np.mean(sqt_h)
        return '{result}'.format(result=calresult)
    except:
        return NoContent, 404

# 计算偏度
def post_skew(array: list):
    try:
        # 偏度 衡量随机分布的不均衡性，偏度 = 0，数值相对均匀的分布在两侧
        calresult = stats.skew(array)
        return '{result}'.format(result=calresult)
    except:
        return NoContent, 404


# 计算峰度
def post_kurtosis(array: list):
    try:
        # 峰度 概率密度在均值处峰值高低的特征
        calresult = stats.kurtosis(array)
        return '{result}'.format(result=calresult)
    except:
        return NoContent, 404



if __name__ == '__main__':
    app = connexion.FlaskApp(__name__, port=8080, specification_dir='swagger/')
    hostname = socket.gethostname()
    app.add_api('solarcalhub-api.yaml', arguments={'title': 'Solar Calculation Hub', 'host': hostname+':9090'})
    app.run()
