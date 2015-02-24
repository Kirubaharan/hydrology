__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import checkdam.checkdam as cd
from datetime import timedelta

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)

aral_rain_file_1 = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/kanaswadi/KSNDMC_01-05-2014_10-09-2014_KANASAWADI.csv'
aral_rain_file_2 = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Kanasawadi_1_09_14_0_02_15.csv'

aral_rain_df_1 = pd.read_csv(aral_rain_file_1, sep=',')
aral_rain_df_1.drop(['TRGCODE', 'DISTRICT', 'TALUKNAME', 'HOBLINAME', 'HOBLICODE', 'Total'], inplace=True, axis=1)
# print aral_rain_df_1.head()
aral_rain_df_2  = pd.read_csv(aral_rain_file_2, sep=',')
aral_rain_df_2.drop(['Sl no', 'TRGCODE', 'HOBLINAME'], inplace=True, axis=1)
print aral_rain_df_2.head()
data_1 = []
for row_no, row in aral_rain_df_1.iterrows():
    date = row['Date']
    for time, value in row.ix[1:, ].iteritems():
        data_1.append((date, time, value))

data_1_df = pd.DataFrame(data_1,columns=['date', 'time', 'rain(mm)'])
data_2 = []
for row_no, row in aral_rain_df_2.iterrows():
    date = row['Date']
    for time, value in row.ix[1:, ].iteritems():
        data_2.append((date, time, value))

data_2_df = pd.DataFrame(data_2,columns=['date', 'time', 'rain(mm)'])
print data_2_df.head()
raise SystemExit(0)
date_format_1 = "%d-%b-%Y %H:%M"
data_1_df['date_time'] = pd.to_datetime(data_1_df['date'] + ' ' + data_1_df['time'], format=date_format_1)
data_1_df.set_index(data_1_df['date_time'], inplace=True)
data_1_df.sort_index(inplace=True)
data_1_df.drop(['date_time', 'date', 'time'], axis=1, inplace=True)

date_format_2 = "%d-%b-%y %H:%M"
data_2_df['date_time'] = pd.to_datetime(data_2_df['date'] + ' ' + data_2_df['time'], format=date_format_2)
data_2_df.set_index(data_2_df['date_time'], inplace=True)
data_2_df.sort_index(inplace=True)
data_2_df.drop(['date_time', 'date', 'time'], axis=1, inplace=True)

# cumulative difference
data_1_8h_df = data_1_df['2014-05-01 8H30T': '2014-09-10 8H30T']
data_1_8h_df['diff'] = 0.000

for d1, d2 in cd.pairwise(data_1_8h_df.index):
    if data_1_8h_df['rain(mm)'][d2] > data_1_8h_df['rain(mm)'][d1]:
        data_1_8h_df['diff'][d2] = data_1_8h_df['rain(mm)'][d2] - data_1_8h_df['rain(mm)'][d1]
        
data_1_30min_df = data_1_8h_df.resample('30Min', how=np.sum, label='right', closed='right')
data_2_8h_df = data_2_df['2014-09-10 8H30T': '2015-02-09 8H30T']

data_2_8h_df['diff'] = 0.000

for d1, d2 in cd.pairwise(data_2_8h_df.index):
    if data_2_8h_df['rain(mm)'][d2] > data_2_8h_df['rain(mm)'][d1]:
        data_2_8h_df['diff'][d2] = data_2_8h_df['rain(mm)'][d2] - data_2_8h_df['rain(mm)'][d1]
        
data_2_30min_df = data_2_8h_df.resample('30Min', how=np.sum, label='right', closed='right')

aral_rain_df = pd.concat([data_1_30min_df, data_2_30min_df], axis=0)


"""
Remove duplicates
"""
aral_rain_df['index'] = aral_rain_df.index
aral_rain_df.drop_duplicates(subset='index', take_last=True, inplace=True)
del aral_rain_df['index']
aral_rain_df = aral_rain_df.sort()
# print aral_rain_df.head()

"""
Weather
"""
weather_ksndmc = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Kanasawadi_weather_01May14_10Feb15.csv'
weather_ksndmc_df = pd.read_csv(weather_ksndmc, sep=',')
weather_ksndmc_df.drop(['Sl no', 'HOBLI'], inplace=True, axis=1)
weather_date_format = "%d-%b-%y %H:%M:%S+05:30"
weather_ksndmc_df['date_time'] = pd.to_datetime(weather_ksndmc_df['DATE'] + " " + weather_ksndmc_df['TIME'], format=weather_date_format)
weather_ksndmc_df.set_index(weather_ksndmc_df['date_time'], inplace=True)
weather_ksndmc_df.sort_index(inplace=True)
cols = weather_ksndmc_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
weather_ksndmc_df = weather_ksndmc_df[cols]
weather_ksndmc_df.drop(['date_time', 'DATE', 'TIME'], inplace=True, axis=1)
weather_ksndmc_df = weather_ksndmc_df.resample('30Min', how=np.mean, label='right', closed='right')
# weather_ksndmc_df = weather_ksndmc_df[ :" 2015-02-09"]
# print weather_ksndmc_df.tail()

base_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/smgoll_1_05_14_09_02_15.csv'
#read csv file
df_base = pd.read_csv(base_file, header=0, sep='\t')
# print df_base.head()
#Drop seconds
df_base['Time'] = df_base['Time'].map(lambda x: x[ :5])
# convert date and time columns into timestamp
# print df_base.head()
date_format = '%d/%m/%y %H:%M'
df_base['Date_Time'] = pd.to_datetime(df_base['Date'] + ' ' + df_base['Time'], format=date_format)
df_base.set_index(df_base['Date_Time'], inplace=True)
# print df_base.columns.values[15]
df_base.columns.values[6] = 'Air Temperature (C)'
df_base.columns.values[7] = 'Min Air Temperature (C)'
df_base.columns.values[8] = 'Max Air Temperature (C)'
df_base.columns.values[15] = 'Canopy Temperature (C)'
df_base['index'] = df_base.index
df_base.drop_duplicates(subset='index', take_last=True, inplace=True)
del df_base['index']
df_base = df_base.sort()
new_index = pd.date_range(start=min(df_base.index), end=max(df_base.index), freq='30min' )
df_base = df_base.reindex(index=new_index, method=None)
df_base = df_base.interpolate(method='time')
weather_df = df_base[['Humidity (%)', 'Solar Radiation (Wpm2)']]
weather_df = weather_df.join(weather_ksndmc_df[['TEMPERATURE', 'WIND_SPEED']])
rain_df = aral_rain_df[['diff']]
rain_df.columns.values[0] = "rain (mm)"
# print rain_df.head()
weather_df.index.name = "Date_Time"
rain_df.index.name = "Date_Time"
weather_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/ksndmc_weather.csv')
rain_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/ksndmc_rain.csv')
# print aral_rain_df.tail()
# print df_base.tail()
# delta = df_base.index[0] - df_base.index[-1]
# print delta.days
# fig= plt.figure()
# plt.plot_date(df_base.index, df_base['Air Temperature (C)'])
# plt.bar(aral_rain_df.index, aral_rain_df['diff'], width=0.02, color='b')
# # plt.bar(df_base.index, df_base['Rain Collection (mm)'],width=0.02, color='g')
# fig.autofmt_xdate(rotation=90)
# plt.show()
# print aral_rain_df['diff'].sum()
# print(df_base['Rain Collection (mm)'].sum())
