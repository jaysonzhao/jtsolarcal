#!/usr/bin/env python3

import socket
import os
import connexion
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import send_from_directory
import scipy.signal as signal
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from k_means import KMeans

# 计算得分value 为实际值 ，expect 是期望值， direction 是方向，也可用于权值，负数为越小越好，正数为越大越好.输入可以是NP ARRAY
def post_calpoint(value: list, expect: list, direction: list):
    calresult = np.sum((np.array(value) - np.array(expect)) * np.array(direction) / np.array(expect))
    return calresult

# 计算kmean预测值
def post_kmeanprict(array: str, centermodel: str, featurename: str):
    testdata = pd.read_json(array, orient='index')
    centers = pd.read_json(centermodel, orient='index')
    columnnames = featurename.split(',')
    testnumber = testdata.shape[0]
    # Get features.
    test_train = testdata[[iaxis for iaxis in columnnames]].values.reshape((testnumber, len(columnnames)))
    centeridmodel = centers[[iaxis for iaxis in columnnames]].values.reshape((len(centers), len(columnnames)))
    closest_centroids_ids = KMeans.centroids_find_closest(test_train, centeridmodel)
    tag = []
    for i in closest_centroids_ids:
        tag.append(centers.index[int(i[0])])

    testdata['tag'] = pd.Series(tag, index=testdata.index)

    return testdata.to_json(orient="index")


# 计算kmean训练值中心值
def post_kmeantrain(array: str, featurename: str):
    data = pd.read_json(array)
    columnnames = featurename.split(',')
    # columnnames = ['DFA', 'violmax', 'maxpeaksqt']
    num_examples = data.shape[0]
    # Get features.
    x_train = data[[iaxis for iaxis in columnnames]].values.reshape((num_examples, len(columnnames)))
    # print(x_train)
    # Set K-Means parameters.
    num_clusters = 4  # Number of clusters into which we want to split our training dataset.
    max_iterations = 50  # maximum number of training iterations.

    # Init K-Means instance.
    k_means = KMeans(x_train, num_clusters)
    # Train K-Means instance.
    (centroids, closest_centroids_ids) = k_means.train(max_iterations)
    # print(centroids)
    data_frame = pd.DataFrame(centroids, columns=[iaxis for iaxis in columnnames])
    #
    dfsort = data_frame.sort_values(by=[columnnames[0]])
    L = [chr(i) for i in range(97, 97 + len(centroids))]
    dfsort['L'] = pd.Series(L, index=dfsort.index)
    dfreturn = dfsort.set_index('L', drop=True)
    # print(dfreturn.to_json(orient="index"))
    return dfreturn.to_json(orient="index")

# 生成拟合曲线
def post_generatelinear(array: list, batch: str, index: str, dpi: int):
    X = np.array(range(0, len(array), 1)).reshape(-1, 1)
    y = np.array(array)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    rmses = []
    degrees = np.arange(1, 10)
    min_rmse, min_deg, score = 1e10, 0, 0
    # print('starting cal')
    for deg in degrees:
        # 生成多项式特征集(如根据degree=3 ,生成 [[x,x**2,x**3]] )
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        x_train_poly = poly.fit_transform(x_train)

        # 多项式拟合
        poly_reg = LinearRegression()
        poly_reg.fit(x_train_poly, y_train)
        # print(poly_reg.coef_,poly_reg.intercept_) #系数及常数

        # 测试集比较
        x_test_poly = poly.fit_transform(x_test)
        y_test_pred = poly_reg.predict(x_test_poly)

        # mean_squared_error(y_true, y_pred) #均方误差回归损失,越小越好。
        poly_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        rmses.append(poly_rmse)
        # r2 范围[0，1]，R2越接近1拟合越好。
        r2score = r2_score(y_test, y_test_pred)

        # degree交叉验证
        if min_rmse > poly_rmse:
            min_rmse = poly_rmse
            min_deg = deg
            # score = r2score
            # print('degree = %s, RMSE = %.2f ,r2_score = %.2f' % (deg, poly_rmse, r2score))

    # 生成多项式特征集(如根据degree=3 ,生成 [[x,x**2,x**3]] )
    poly = PolynomialFeatures(degree=min_deg, include_bias=False)
    x_poly = poly.fit_transform(X)

    # 多项式拟合
    poly_reg = LinearRegression()
    poly_reg.fit(x_poly, y)

    newy = poly_reg.predict(poly.fit_transform(X))

    fig = plt.figure()  # 实例化作图变量
    plt.title(batch + ',' + index)  # 图像标题
    plt.xlabel('x')  # x轴文本
    plt.ylabel('y')  # y轴文本
    plt.plot(X, newy, 'k.')
    if not os.path.exists('graph'):
        os.mkdir('graph')
    fig.savefig('graph\\' + batch + index + 'linear.png', format='png', dpi=dpi)
    plt.close()
    return batch + index + 'linear.png'


# 生成离散折线
def post_generategraph(array: list, batch: str, index: str, dpi: int):
    X = np.array(range(0, len(array), 1)).reshape(-1, 1)
    y = np.array(array)
    peakidx, _ = signal.find_peaks(y, prominence=1)
    fig = plt.figure()  # 实例化作图变量
    plt.title(batch + ',' + index)  # 图像标题
    plt.xlabel('x')  # x轴文本
    plt.ylabel('y')  # y轴文本
    plt.plot(X, y, linewidth=1, color='blue', marker='o', markerfacecolor='black', markersize=3)
    plt.plot(peakidx, y[peakidx], color='red', label='peaks')
    if not os.path.exists('graph'):
        os.mkdir('graph')
    fig.savefig('graph\\' + batch + index + 'plot.png', format='png', dpi=dpi)
    plt.close()
    return batch + index + 'plot.png'


app = connexion.FlaskApp(__name__, port=9091, specification_dir='swagger/')


@app.route('/graph/<path:filename>')
def download_file(filename):
    return send_from_directory('graph\\',
                               filename, as_attachment=False)


if __name__ == '__main__':
    hostname = socket.gethostname()
    app.add_api('solarcalsupp-api.yaml',
                arguments={'title': 'Solar Calculation Supplements', 'host': hostname + ':9091'})
    app.run()
