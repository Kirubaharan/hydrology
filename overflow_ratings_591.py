__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

cd_wall_length_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/ch_591_wall_elevation.csv'
cd_wall_length_df = pd.read_csv(cd_wall_length_file, sep=',', header=0)
cd_wall_length_df.set_index(cd_wall_length_df['elevation'], inplace=True)
print cd_wall_length_df
height = np.arange(0.0, 1.01, 0.01)
g = 9.81  # m2/s
width_check_dam = 0.55   # t
cd_1 = 1.45 # 0.3 h/t ratio
cd_2 = 1.822 # 3 h/t ratio



