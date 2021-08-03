import numpy as np
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)


# 从polyman_data（通道数据）和annotation_data.txt（注释数据）中分别提取np_wave(Nx3000)和np_anno(Nx1)
# 需要两个参数：stage_on和stage_off，根据annotation_data.txt中的各个sleep stage（最后一列）提取某一段data信息
def extract_data(stage_on, stage_off):
    np_wave = np.loadtxt('E:/PSG/data_label/polyman_data', encoding='utf-8', delimiter=';', skiprows=1, usecols=4)
    df_anno = pd.read_csv('E:/PSG/data_label/annotation_data.txt', encoding='ansi')

    start = int(df_anno.loc[stage_on, ' Recording onset'] / 30)
    end = int(df_anno.loc[stage_off, ' Recording onset'] / 30 - 1)
    N = end - start + 1

    np_wave = np_wave.reshape((-1, 3000))
    np_wave = np_wave[start:end+1, :]
    # Nx3000的输入矩阵
    output = pd.DataFrame(range(N))
    np_anno = np.zeros((N, 1))
    # Nx1的输出矩阵
    # 4: R期    0: W期     1: 1期......
    step = 30
    wave_index = 0
    for anno_row in range(stage_on, stage_off):
        add_index = int(df_anno.loc[anno_row, ' Duration']) / step
        output[int(wave_index):int(wave_index + add_index)] = df_anno.loc[anno_row, ' Annotation']
        wave_index += add_index
    for i in range(len(output)):
        if output.loc[i, 0] == ' Sleep stage 1':
            np_anno[i, 0] = 1
        elif output.loc[i, 0] == ' Sleep stage 2':
            np_anno[i, 0] = 2
        elif output.loc[i, 0] == ' Sleep stage 3' or output.loc[i, 0] == ' Sleep stage 4':
            np_anno[i, 0] = 3
        elif output.loc[i, 0] == ' Sleep stage R':
            np_anno[i, 0] = 4
    return np_wave, np_anno.astype(int)

# 以下为使用演示：
# stage_on=1表示从第二个sleep stage开始，stage_off=6表示到第7个sleep stage结束（不包含），之间经历了24帧（24x30s）的时间
x_data, y_data = extract_data(1, 6)
with open('E:/PSG/data_label/partial.txt', 'w') as fx:

    for i in range(len(x_data)):

        for j in range(3000):
            fx.write(str(x_data[i][j])+',')

        fx.write(str(y_data[i][0])+'\n')
