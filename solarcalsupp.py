#!/usr/bin/env python3

import socket

import connexion
import numpy as np


# 计算得分value 为实际值 ，expect 是期望值， direction 是方向，也可用于权值，负数为越小越好，正数为越大越好.输入可以是NP ARRAY
def post_calpoint(value: list, expect: list, direction: list):
    calresult = np.sum((np.array(value) - np.array(expect)) * np.array(direction) / np.array(expect))
    return calresult


if __name__ == '__main__':
    app = connexion.FlaskApp(__name__, port=9091, specification_dir='swagger/')
    hostname = socket.gethostname()
    app.add_api('solarcalsupp-api.yaml', arguments={'title': 'Solar Calculation Supplements', 'host': hostname})
    app.run()
