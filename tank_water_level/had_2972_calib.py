__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import checkdam.checkdam as cd


"""
Calibration - Data logger No: 2972
"""

y_cal = np.array([100, 500, 800, 1200, 1800, 2400, 3000, 3600, 3700])
x_cal = np.array([1901, 2176, 2393, 2668, 3095, 3496, 3914, 4330, 4403])
a_stage = cd.polyfit(x_cal, y_cal, 1)
coeff_cal = a_stage['polynomial']
slope = coeff_cal[0]
intercept = coeff_cal[1]
"""
Read Check dam data
"""
block_1 = '/media/kiruba/New Volume/ACCUWA_Data/lake_water_level/2972/2972_010_001.CSV'
water_level_1 = cd.read_correct_ch_dam_data(block_1, slope, intercept)
block_2 = '/media/kiruba/New Volume/ACCUWA_Data/lake_water_level/2972/2972_010_002.CSV'
water_level_2 = cd.read_correct_ch_dam_data(block_2, slope, intercept)
block_3 = '/media/kiruba/New Volume/ACCUWA_Data/lake_water_level/2972/2972_015_001.CSV'
water_level_3 = cd.read_correct_ch_dam_data(block_3, slope, intercept)
block_4 = '/media/kiruba/New Volume/ACCUWA_Data/lake_water_level/2972/2972_015_002_22_08_2015.CSV'
water_level_4 = cd.read_correct_ch_dam_data(block_4, slope, intercept)
block_5 = '/media/kiruba/New Volume/ACCUWA_Data/lake_water_level/2972/2972_007_001_24_12_2015.CSV'
water_level_5 = cd.read_correct_ch_dam_data(block_5, slope, intercept)

# 2015-09-10 12:00:00
water_level_2.drop('2015-09-10 12:00:00', inplace=True, axis=0)
print water_level_1.head()
print water_level_1.tail()
print water_level_2.head()
print water_level_2.tail()
print water_level_3.head()
print water_level_3.tail()
print water_level_4.head()
print water_level_4.tail()
print water_level_5.head()
print water_level_5.tail()

for i in range(1, 6, 1):
    eval("water_level_{0}.drop(water_level_{0}.tail(1).index, inplace=True, axis=0)".format(i))
    eval("water_level_{0}.drop(water_level_{0}.head(1).index, inplace=True, axis=0)".format(i))

fig = plt.figure()
for i in range(1, 6, 1):
    x = eval("water_level_{0}.index".format(i))
    y = eval("water_level_{0}['stage(m)']".format(i))
    plt.plot(x, y)

plt.show()

water_level = pd.concat([water_level_1, water_level_2, water_level_3, water_level_4, water_level_5], axis=0)
water_level.sort_index(inplace=True)
print water_level.tail()
water_level_bathymetry_verify_df = water_level['2015-12-24']
print water_level_bathymetry_verify_df.head()
water_level.to_csv('/media/kiruba/New Volume/Mail/Sierra/water_level_had_tank.csv')
water_level_bathymetry_verify_df.to_csv('/media/kiruba/New Volume/Mail/Sierra/water_level_had_tank_24_12_15.csv')