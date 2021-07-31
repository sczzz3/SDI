import pandas as pd
import numpy as np


def create_data():
    sep = 0
    idx = 0
    x = []
    y = []
    data = pd.read_csv('/Users/aoao/Desktop/result', encoding='utf-8')
    for i in range(1, 7950001, 3000):
        x.append(np.array(data.loc[i:i+2999, 'EEG Fpz-Cz[uV]']).tolist())
        y.append(data.loc[i, 'Annotation'])
    num = int(7950000 / 3000)
    with open('/Users/aoao/Desktop/x', 'w') as fx:
        for i in range(num):
            fx.write(str(x[i]))
            fx.write('\n')
    with open('/Users/aoao/Desktop/y', 'w') as fy:
        for i in range(num):
            fy.write(y[i][-1])
            fy.write('\n')


create_data()

