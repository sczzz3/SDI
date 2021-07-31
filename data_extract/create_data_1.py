# This program generates correct stage labels in an file exported from .edf
import pandas as pd


def main():
    df_wave = pd.read_csv('/Users/aoao/Desktop/SC4002E0-PSG.edf', encoding='utf-8', sep=';')
    df_wave['Annotation'] = 'Sleep stage W'
    df_anno = pd.read_csv('/Users/aoao/Desktop/SC4002EC-Hypnogram.edf', encoding='gbk')
    step = 0.01
    wave_row = 0
    for anno_row in range(len(df_anno)):
        new_rows = int(df_anno.loc[anno_row, ' Duration'])/step
        df_wave.loc[wave_row+1:wave_row+new_rows, 'Annotation'] = df_anno.loc[anno_row, ' Annotation']
        wave_row += new_rows
    df_wave.to_csv('/Users/aoao/Desktop/test', header=True)


main()
