__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import checkdam.checkdam as cd

# rain file
rain_file = '/media/kiruba/New Volume/KSNDMC 15 mins Daily Data/dailyrainfalldata15minsdailyrainfalldata15minsf/TUBAGERE.csv'
rain_df = pd.read_csv(rain_file, sep=',', header=0)
# print rain_df.head()
rain_df.drop(["TRGCODE", "DISTRICT", "TALUKNAME", "HOBLINAME", "HOBLICODE", "PHASE", "COMPANY", "TYPE", "CATEGORY", "FIRSTREPORTED", "Total" ], inplace=True, axis=1)

data_1 = []
for row_no, row in rain_df.iterrows():
    date = row['Date']
    for time, value in row.ix[1:, ].iteritems():
        data_1.append((date, time, value))

data_1_df = pd.DataFrame(data_1,columns=['date', 'time', 'rain(mm)'])
# print data_1_df.head()
# print data_1_df.tail()
date_format_1 = "%d-%b-%y %H:%M"
data_1_df['date_time'] = pd.to_datetime(data_1_df['date'] + ' ' + data_1_df['time'], format=date_format_1)
data_1_df.set_index(data_1_df['date_time'], inplace=True)
data_1_df.sort_index(inplace=True)
data_1_df.drop(['date_time', 'date', 'time'], axis=1, inplace=True)

# cumulative difference
data_1_8h_df = data_1_df['2010-01-01 8H30T': '2015-11-30 8H30T']
data_1_8h_df['diff'] = 0.000

for d1, d2 in cd.pairwise(data_1_8h_df.index):
    if data_1_8h_df['rain(mm)'][d2] > data_1_8h_df['rain(mm)'][d1]:
        data_1_8h_df['diff'][d2] = data_1_8h_df['rain(mm)'][d2] - data_1_8h_df['rain(mm)'][d1]

"""
Remove duplicates
"""
rain_df = data_1_8h_df
rain_df['index'] = rain_df.index
rain_df.drop_duplicates(subset='index', take_last=True, inplace=True)
del rain_df['index']
rain_df.sort_index(inplace=True)
# print rain_df.head()

# resample_daily
rain_df_daily_had = rain_df.resample('D', how=np.sum, label='left', closed='left')
print rain_df_daily_had.head()
rain_df_daily_had.to_csv('/media/kiruba/New Volume/milli_watershed/cumulative impacts/had_rainfall_daily.csv')
rain_df.to_csv('/media/kiruba/New Volume/milli_watershed/cumulative impacts/had_rainfall.csv')