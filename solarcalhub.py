#!/usr/bin/env python3

import connexion
from connexion import NoContent
import numpy as np

#计算平均值
def post_mean(array: list):
        try:
            # 求均值
            calresult = np.mean(array)
            return '{result}'.format(result=calresult)
        except:
            return NoContent, 404


#计算变化率标准差
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


if __name__ == '__main__':
    app = connexion.FlaskApp(__name__, port=9090, specification_dir='swagger/')
    app.add_api('solarcalhub-api.yaml', arguments={'title': 'Solar Calculation Hub'})
    app.run()
