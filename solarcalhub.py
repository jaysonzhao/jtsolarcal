#!/usr/bin/env python3

import connexion
import numpy as np


def post_greeting(name: str) -> str:
    inputstr = str.split(",")
    arr = []
    for numi in inputstr:
        try:
            arr.append(float(numi))
        except ValueError:
            continue
    # 求均值
    name = np.mean(arr)
    return '{name}'.format(name=name)

if __name__ == '__main__':
    app = connexion.FlaskApp(__name__, port=9090, specification_dir='swagger/')
    app.add_api('helloworld-api.yaml', arguments={'title': 'Hello World Example'})
    app.run()
