#!/usr/bin/env python3

import connexion
from connexion import NoContent
import numpy as np

#计算平均值
def post_mean(array: str) -> str:
    inputstr = array.split(",")
    arr = []
    for numi in inputstr:
        try:
            arr.append(float(numi))
        except ValueError:
            return NoContent, 404
    # 求均值
    calresult = np.mean(arr)
    return '{result}'.format(result=calresult)

#计算变化率标准差
def post_rocsd(array: str) -> str:
    inputstr = array.split(",")
    arr = []
    for numi in inputstr:
        try:
            arr.append(float(numi))
        except ValueError:
            return NoContent, 404
    # 计算变化率
    result = []
    it = iter(arr)
    n1 = float(it.__next__())
    for n2 in it:
        result.append((n2 - n1) / n1)
        n1 = float(n2)
    # 求变化率标准差
    calresult = np.std(result, ddof=1)
    return '{result}'.format(result=calresult)

if __name__ == '__main__':
    app = connexion.FlaskApp(__name__, port=9090, specification_dir='swagger/')
    app.add_api('solarcalhub-api.yaml', arguments={'title': 'Solar Calculation Hub'})
    app.run()
