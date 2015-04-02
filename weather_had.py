__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import checkdam.checkdam as cd
from datetime import timedelta
import datetime
import pymc as pm
from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform, Lambda, MCMC, observed, poisson_like
from pymc.distributions import Impute
from scipy.stats import itemfreq, norm
import scipy.stats as stats

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=18)

# aral_rain_file_1 = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/kanaswadi/KSNDMC_01-05-2014_10-09-2014_KANASAWADI.csv'
aral_rain_file_1 = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tubgere_1_09_14_10_02_15.csv'

aral_rain_df_1 = pd.read_csv(aral_rain_file_1, sep=',', header=0)
aral_rain_df_1.drop(['Sl no', 'TRGCODE', 'HOBLINAME'], inplace=True, axis=1)
# print aral_rain_df_1.head()
# raise SystemExit(0)
# aral_rain_df_2  = pd.read_csv(aral_rain_file_2, sep=',')
# aral_rain_df_2.drop(['Sl no', 'TRGCODE', 'HOBLINAME'], inplace=True, axis=1)
# print aral_rain_df_2.head()
data_1 = []
for row_no, row in aral_rain_df_1.iterrows():
    date = row['Date']
    for time, value in row.ix[1:, ].iteritems():
        data_1.append((date, time, value))

data_1_df = pd.DataFrame(data_1,columns=['date', 'time', 'rain(mm)'])


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
        
data_1_30min_df = data_1_8h_df.resample('30Min', how=np.sum, label='right', closed='right')
aral_rain_df = data_1_30min_df

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
weather_ksndmc = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tubgere_weather_corrected_wind_speed.csv'
weather_ksndmc_df = pd.read_csv(weather_ksndmc, sep=',')
print weather_ksndmc_df.tail()
# weather_ksndmc_df.drop(['Sl no', 'HOBLI'], inplace=True, axis=1)
weather_date_format = "%Y-%m-%d %H:%M:%S"
weather_ksndmc_df['Date_Time'] = pd.to_datetime(weather_ksndmc_df['Date_Time'], format=weather_date_format)
weather_ksndmc_df.set_index(weather_ksndmc_df['Date_Time'], inplace=True)
weather_ksndmc_df.sort_index(inplace=True)
cols = weather_ksndmc_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
weather_ksndmc_df = weather_ksndmc_df[cols]
# weather_ksndmc_df.drop(['date_time', 'DATE', 'TIME'], inplace=True, axis=1)
# start_time = min(weather_ksndmc_df.index)
# end_time = max(weather_ksndmc_df.index)
# new_index = pd.date_range(start=start_time, end=end_time, freq='15min')
# weather_ksndmc_df = weather_ksndmc_df.reindex(new_index, method=None)
# weather_ksndmc_df = weather_ksndmc_df.interpolate(method="time")
# weather_ksndmc_df['WIND_SPEED'].apply(pd.Series.interpolate)
# weather_ksndmc_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tubgere_weather_01May14_10Feb15_corrected.csv')
# weather_ksndmc_df['WIND_SPEED'] = weather_ksndmc_df['WIND_SPEED'].astype(float)
# print weather_ksndmc_df.tail()
print weather_ksndmc_df.dtypes
# fig = plt.figure()
# plt.plot(weather_ksndmc_df.index, weather_ksndmc_df["WIND_SPEED"])
# plt.show()
"""
Include temp and humidity data from hadonahalli weather station data
"""
# weather_had_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/corrected_weather.csv'
# weather_had_df = pd.read_csv(weather_had_file, sep=',')
# print weather_had_df.head()
# raise SystemExit(0)

# weather_ksndmc_df = weather_ksndmc_df.interpolate(method='time')
# print weather_ksndmc_df['2014-06-04 07:00:00':]
weather_ksndmc_df = weather_ksndmc_df.resample('30Min', how=np.mean, label='right', closed='right')
# weather_regr_df = weather_ksndmc_df[:'2014-05-17']
# weather_ksndmc_df = weather_ksndmc_df[ :" 2015-02-09"]
# print weather_ksndmc_df.tail()

base_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/had_june_jan_10.csv'
may_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/Hadonahalli_WS_May 2014.csv'
#read csv file
df_base = pd.read_csv(base_file, header=0, sep=',')
may_df = pd.read_csv(may_file, header=0, sep=',')
# may_df['Time'] = may_df['Time'].map(lambda x: x[ :5])
# print may_df.head()
#Drop seconds
df_base['Time'] = df_base['Time'].map(lambda x: x[ :5])
# convert date and time columns into timestamp
# print df_base.head()
date_format = '%d/%m/%y %H:%M'
df_base['Date_Time'] = pd.to_datetime(df_base['Date'] + ' ' + df_base['Time'], format=date_format)
df_base.set_index(df_base['Date_Time'], inplace=True)
date_format = '%d/%m/%y %H:%M:%S'
# print may_df['Date'][0] + ' ' + may_df['Time'][0]
may_df['Date_Time'] = pd.to_datetime(may_df['Date'] + ' ' +  may_df["Time"], format=date_format)
may_df.set_index(may_df['Date_Time'], inplace=True)
rounded = np.array(may_df.index, dtype='datetime64[m]')
may_df.set_index(rounded,inplace=True)
may_df.columns.values[6] = 'Air Temperature (C)'
may_df.columns.values[7] = 'Min Air Temperature (C)'
may_df.columns.values[8] = 'Max Air Temperature (C)'
may_df.columns.values[15] = 'Canopy Temperature (C)'
may_df['index'] = may_df.index
may_df.drop_duplicates(subset='index', take_last=True, inplace=True)
del may_df['index']
may_df = may_df.sort()
new_index = pd.date_range(start=min(may_df.index), end=max(may_df.index), freq='30min' )
may_df = may_df.reindex(index=new_index, method=None)
may_df = may_df.interpolate(method='time')
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
df_base = pd.concat([may_df, df_base], axis=0)
print df_base.columns.values
print max(df_base.index)
print max(weather_ksndmc_df.index)
# print weather_ksndmc_df.tail()
df_base['Wind Speed (mps)'] = weather_ksndmc_df['WIND_SPEED'][df_base.index]
print df_base.tail()
fig = plt.figure()
plt.plot(df_base.index, df_base['Wind Speed (mps)'], 'r-')
plt.show()


rain_df = aral_rain_df[['diff']]
# rain_df.columns.values[0] = "Rain Collection (mm)"
rain_df_1 = df_base['Rain Collection (mm)'][ : '2014-09-01 8H00T']
rain_df_1 = pd.DataFrame([rain_df_1[i] for i in range(len(rain_df_1))], columns=['diff'], index=rain_df_1.index)
rain_df = pd.concat([rain_df_1, rain_df], axis=0)
rain_df.columns.values[0] = "rain (mm)"


df_base.index.name = "Date_Time"
rain_df.index.name = "Date_Time"
df_base.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/corrected_weather_ws.csv')
rain_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/ksndmc_rain.csv')
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
# print(df_base['Rain Collection (mm)'].