__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import checkdam.checkdam as cd
from matplotlib import rc
import matplotlib.cm as cmx
import matplotlib.colors as colors
from datetime import timedelta
import math
import ccy_classic_lstsqr

date_format = '%Y-%m-%d %H:%M:%S'
daily_format = '%Y-%m-%d'

y_cal = np.array([100, 400, 1000, 1600, 2250, 2750])
x_cal = np.array([1987, 2454, 3344, 4192, 5104, 5804])
a_stage = cd.polyfit(x_cal, y_cal, 1)
coeff_cal = a_stage['polynomial']
slope = coeff_cal[0]
intercept = coeff_cal[1]

block_1 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2526/2526_007_001.CSV'
water_level_1 = cd.read_correct_ch_dam_data(block_1, slope, intercept)
block_2 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2526/2526_007_002_25_8_14.CSV'
water_level_2 = cd.read_correct_ch_dam_data(block_2, slope, intercept)
block_3 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2526/2526_007_003.CSV'
water_level_3 = cd.read_correct_ch_dam_data(block_3, slope, intercept)
block_4 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2526/2526_007_004.CSV'
water_level_4 = cd.read_correct_ch_dam_data(block_4, slope, intercept)
block_5 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2526/2526_005_001.CSV'
water_level_5 = cd.read_correct_ch_dam_data(block_5, slope, intercept)
block_6 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2526/2526_005_002_03_12_2014.CSV'
water_level_6 = cd.read_correct_ch_dam_data(block_6, slope, intercept)

fig = plt.figure()
for i in range(1, 7, 1):
    x = eval("water_level_{0}.index".format(i))
    y = eval("water_level_{0}['stage(m)']".format(i))
    plt.plot(x, y)

plt.show()