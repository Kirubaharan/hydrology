__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import checkdam.checkdam as cd

date_format = '%Y-%m-%d %H:%M:%S'
"""
634
"""
stage_634_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/stage_634.csv'
stage_634_df = pd.read_csv(stage_634_file,sep=',')
# print stage_634_df.head()
stage_634_df['date_time'] = pd.to_datetime(stage_634_df['date_time'], format=date_format)
stage_634_df.set_index(stage_634_df['date_time'], inplace=True)
"""
625
"""
stage_625_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_625/stage_625.csv'
stage_625_df = pd.read_csv(stage_625_file, sep=',')
stage_625_df['Date'] = pd.to_datetime(stage_625_df['Date'], format=date_format)
stage_625_df.set_index(stage_625_df['Date'], inplace=True)
"""
463
"""
stage_463_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/water_level.csv'
stage_463_df = pd.read_csv(stage_463_file, sep=',')
stage_463_df['Date'] = pd.to_datetime(stage_463_df['Date'], format=date_format)
stage_463_df.set_index(stage_463_df['Date'], inplace=True)
"""
591
"""
stage_591_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/stage_591.csv'
stage_591_df = pd.read_csv(stage_591_file, sep=',')
stage_591_df['Date'] = pd.to_datetime(stage_591_df['Date'], format=date_format)
stage_591_df.set_index(stage_591_df['Date'], inplace=True)
"""
599
"""
stage_599_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_599/stage_599.csv'
stage_599_df = pd.read_csv(stage_599_file, sep=',')
print stage_599_df.head()
stage_599_df['Date'] = pd.to_datetime(stage_599_df['Date'], format=date_format)
stage_599_df.set_index(stage_599_df['Date'], inplace=True)
# Aralumallige
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(stage_591_df.index, stage_591_df['stage(m)'], 'bo')
ax2.plot(stage_599_df.index, stage_599_df['stage(m)'], 'ro')
ax1.set_title("Check dam 591")
ax2.set_title("Check dam 599")
plt.show()
# Hadonahalli
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
ax1.plot(stage_634_df.index, stage_634_df['stage(m)'], 'bo')
ax2.plot(stage_625_df.index, stage_625_df['stage(m)'], 'ro')
ax3.plot(stage_463_df.index, stage_463_df['stage(m)'], 'go')
ax1.set_title("Check dam 634")
ax2.set_title("Check dam 625")
ax3.set_title("Check dam 463")
plt.show()