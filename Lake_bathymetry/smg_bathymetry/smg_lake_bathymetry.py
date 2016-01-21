__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import checkdam.checkdam as cd


# calibration
y_cal = np.array([100, 1000, 2000, 3000, 4000, 5000])
x_cal = np.array([1875, 2516, 3212, 3901, 4605, 5280])
a_stage = cd.polyfit(x_cal, y_cal, 1)
coeff_cal = a_stage['polynomial']
slope = coeff_cal[0]
interecept = coeff_cal[1]
# depth correction based on capacitance
block_1 = '/media/kiruba/New Volume/ACCUWA_Data/lake_water_level/3055/3055_012_002_21_08_2015.CSV'
block_2 = '/media/kiruba/New Volume/ACCUWA_Data/lake_water_level/3055/3055_015_001.CSV'
block_3 = '/media/kiruba/New Volume/ACCUWA_Data/lake_water_level/3055/3055_010_001_08_01_2015.CSV'
block_1_df = cd.read_correct_ch_dam_data(block_1, calibration_slope=slope, calibration_intercept=interecept)
block_2_df = cd.read_correct_ch_dam_data(block_2, calibration_slope=slope, calibration_intercept=interecept)
block_3_df = cd.read_correct_ch_dam_data(block_3, calibration_slope=slope, calibration_intercept=interecept)
fig = plt.figure()
plt.plot(block_1_df.index, block_1_df['stage(m)'], 'g-')
plt.plot(block_2_df.index, block_2_df['stage(m)'], 'g-')
plt.plot(block_3_df.index, block_3_df['stage(m)'], 'g-')
plt.show()