__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import itertools
import checkdam.checkdam as cd
"""
Capacitance sensor Calibration
"""
# 2972
y_cal = np.array([100, 500, 800, 1200, 1800, 2400, 3000, 3600, 3700])
x_cal = np.array([1901, 2176, 2393, 2668, 3095, 3496, 3914,4330,4403])
a_stage = cd.polyfit(x_cal, y_cal, 1)
coeff_cal = a_stage['polynomial']
slope = coeff_cal[0]
intercept = coeff_cal[1]
print coeff_cal
"""
read tank data
"""
block_1 = '/media/kiruba/New Volume/ACCUWA_Data/lake_water_level/2972/2972_015_001.CSV'
water_level_1 = cd.read_correct_ch_dam_data(block_1, slope, intercept)
block_2 = '/media/kiruba/New Volume/ACCUWA_Data/lake_water_level/2972/2972_015_002_22_08_2015.CSV'
water_level_2 = cd.read_correct_ch_dam_data(block_2, slope, intercept)
block_3 = '/media/kiruba/New Volume/ACCUWA_Data/lake_water_level/2972/2972_010_001.CSV'
water_level_3 = cd.read_correct_ch_dam_data(block_3, slope, intercept)
block_4 = '/media/kiruba/New Volume/ACCUWA_Data/lake_water_level/2972/2972_010_002.CSV'
water_level_4 = cd.read_correct_ch_dam_data(block_4, slope, intercept)
block_5 = '/media/kiruba/New Volume/ACCUWA_Data/lake_water_level/2972/2972_007_001_24_12_2015.CSV'
water_level_5 = cd.read_correct_ch_dam_data(block_5, slope, intercept)

# remove error value when logger is removed for downloading
# http://stackoverflow.com/a/16613835/2632856
water_level_4.drop(pd.Timestamp('2015-09-10 12:00:00'), inplace=True, axis=0)
# drop last and first value
for i in range(1, 6, 1):
    eval("water_level_{0}.drop(water_level_{0}.tail(1).index, inplace=True, axis=0)".format(i))
    eval("water_level_{0}.drop(water_level_{0}.head(1).index, inplace=True, axis=0)".format(i))

fig = plt.figure()
for i in range(1, 6, 1):
    x = eval("water_level_{0}.index".format(i))
    y = eval("water_level_{0}['stage(m)']".format(i))
    plt.plot(x, y, label='water_level_{0}'.format(i))

plt.legend()
plt.show()
# # print water_level['2015-09-10 09:40:00': '2015-09-10 14:10:00']
# for i in range(1, 6, 1):
#     print "water_level_{0}".format(i)
#     print eval("water_level_{0}.head()".format(i))
#     print eval("water_level_{0}.tail()".format(i))

water_level_30min = pd.concat([water_level_1, water_level_2], axis=0)
water_level_30min.sort_index(inplace=True)
start_time_30 = min(water_level_30min.index)
end_time_30 = max(water_level_30min.index)
new_index_30 = pd.date_range(start=start_time_30, end=end_time_30, freq='30min')
water_level_30min = water_level_30min.reindex(new_index_30, method=None)
water_level_30min = water_level_30min.interpolate(method='time')

water_level_10min = pd.concat([water_level_3, water_level_4, water_level_5], axis=0)
water_level_10min.sort_index(inplace=True)
start_time_10 = min(water_level_10min.index)
end_time_10 = max(water_level_10min.index)
new_index_10 = pd.date_range(start=start_time_10, end=end_time_10, freq='10min')
water_level_10min = water_level_10min.reindex(new_index_10, method=None)
water_level_10min = water_level_10min.interpolate(method='time')


water_level = pd.concat([water_level_30min, water_level_10min], axis=0)
water_level = water_level.resample('30min', how=np.mean, label='right', closed='right')
water_level.sort_index(inplace=True)
print water_level.head()
water_level.index.name = 'date_time'
water_level.to_csv('/media/kiruba/New Volume/milli_watershed/cumulative impacts/tmg_lake/stage_tmg.csv')