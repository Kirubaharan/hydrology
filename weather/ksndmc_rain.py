__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

# rain file
rain_file = '/media/kiruba/New Volume/KSNDMC 15 mins Daily Data/dailyrainfalldata15minsdailyrainfalldata15minsf/KANASAWADI.csv'
rain_df = pd.read_csv(rain_file, sep=',', header=0)
# print rain_df.head()
rain_df.drop(["TRGCODE", "DISTRICT", "TALUKNAME", "HOBLINAME", "HOBLICODE", "PHASE", "COMPANY", "TYPE", "CATEGORY", "FIRSTREPORTED", "Total" ], inplace=True, axis=1)

data_1 = []
for row_no, row in rain_df.iterrows():
    date = row['Date']
    for time, value in row.ix[1:, ].iteritems():
        data_1.append((date, time, value))

data_1_df = pd.DataFrame(data_1,columns=['date', 'time', 'rain(mm)'])
print data_1_df.head()
date_format_1 = "%d-%b-%y %H:%M"
data_1_df['date_time'] = pd.to_datetime(data_1_df['date'] + ' ' + data_1_df['time'], format=date_format_1)
data_1_df.set_index(data_1_df['date_time'], inplace=True)
data_1_df.sort_index(inplace=True)
data_1_df.drop(['date_time', 'date', 'time'], axis=1, inplace=True)

# cumulative difference
data_1_8h_df = data_1_df['2014-09-01 8H30T': '2015-02-09 8H30T']
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
# print rain_df.tail()

