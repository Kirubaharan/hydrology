__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from bokeh.plotting import figure, show, output_file, gridplot
import checkdam.checkdam as cd

date_format = '%d/%m/%y %H:%M:%S'
daily_format = '%d/%m/%y %H:%M'
# raise SystemExit(0)
# hadonahalli weather station
weather_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/had_may_14_feb_16.csv'
weather_df = pd.read_csv(weather_file, sep='\t', header=0, encoding='utf-8')
# raise SystemExit(0)
weather_df.columns.values[6] = 'Air Temperature (C)'
weather_df.columns.values[7] = 'Min Air Temperature (C)'
weather_df.columns.values[8] = 'Max Air Temperature (C)'
weather_df.columns.values[15] = 'Canopy Temperature (C)'
# raise SystemExit(0)
weather_df['date_time'] = pd.to_datetime(weather_df['Date'] + ' ' + weather_df['Time'], format=date_format)

weather_df = weather_df.set_index(weather_df['date_time'])
weather_df.drop(['Date',
                 'Time',
                 "date_time",
                 'Rain Collection (mm)',
                 'Barometric Pressure (KPa)',
                 'Soil Moisture',
                 'Leaf Wetness',
                 'Canopy Temperature (C)',
                 'Evapotranspiration',
                 'Charging status',
                 'Solar panel voltage',
                 'Network strength',
                 'Battery strength'], axis=1, inplace=True)

"""
Remove Duplicates
"""
# # print df_base.count()

# print df_base.head()
# print df_base.count()

weather_df['index'] = weather_df.index
weather_df.drop_duplicates(subset='index', keep='last', inplace=True)
del weather_df['index']

weather_df.sort_index(inplace=True)

# raise SystemExit(0)

weather_df.loc[(weather_df['Humidity (%)'] < 12) | (weather_df['Min Air Temperature (C)'] < 10) | (weather_df['Max Air Temperature (C)'] > 40) | (weather_df['Wind Speed (kmph)'] == 0.0) | (weather_df['Wind Speed (kmph)'] > 12), 'Wind Speed (kmph)'] = np.nan
weather_df.loc[(weather_df['Humidity (%)'] < 12) | (weather_df['Min Air Temperature (C)'] < 10) | (weather_df['Max Air Temperature (C)'] > 40) | (weather_df['Wind Speed (kmph)'] == 0.0) | (weather_df['Wind Speed (kmph)'] > 12), 'Max Air Temperature (C)'] = np.nan
weather_df.loc[(weather_df['Humidity (%)'] < 12) | (weather_df['Min Air Temperature (C)'] < 10) | (weather_df['Max Air Temperature (C)'] > 40) | (weather_df['Wind Speed (kmph)'] == 0.0) | (weather_df['Wind Speed (kmph)'] > 12), 'Min Air Temperature (C)'] = np.nan
weather_df.loc[(weather_df['Humidity (%)'] < 12) | (weather_df['Min Air Temperature (C)'] < 10) | (weather_df['Max Air Temperature (C)'] > 40) | (weather_df['Wind Speed (kmph)'] == 0.0) | (weather_df['Wind Speed (kmph)'] > 12), 'Humidity (%)'] = np.nan
weather_df.loc[(weather_df['Humidity (%)'] < 12) | (weather_df['Min Air Temperature (C)'] < 10) | (weather_df['Max Air Temperature (C)'] > 40) | (weather_df['Wind Speed (kmph)'] == 0.0) | (weather_df['Wind Speed (kmph)'] > 12), 'Solar Radiation (Wpm2)'] = np.nan
# weather_df.loc[weather_df['Humidity (%)'] < 12, 'Humidity (%)'] = np.nan
weather_df.loc[:, 'Air Temperature (C)'] = 0.5*(weather_df.loc[:, 'Max Air Temperature (C)'] + weather_df.loc[:,'Min Air Temperature (C)'])
print weather_df.head()

# resample and interpolate
weather_df = weather_df.resample('30Min', how=np.mean, label='right', closed='right')
start_time = min(weather_df.index)
end_time = max(weather_df.index)
new_index = pd.date_range(start=start_time, end=end_time, freq='30Min')
weather_df = weather_df.reindex(new_index, method=None)
# weather_df = weather_df.interpolate(method='barycentric')
# weather_df = weather_df.fillna(method='ffill')
#  temperature, windspeed, solar radiaiton, humidity
#
# weather_ksndmc = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tubgere_weather_corrected_wind_speed.csv'
# weather_ksndmc_df = pd.read_csv(weather_ksndmc, sep=',')
# print weather_ksndmc_df.tail()
# # weather_ksndmc_df.drop(['Sl no', 'HOBLI'], inplace=True, axis=1)
# weather_date_format = "%Y-%m-%d %H:%M:%S"
# weather_ksndmc_df['Date_Time'] = pd.to_datetime(weather_ksndmc_df['Date_Time'], format=weather_date_format)
# weather_ksndmc_df.set_index(weather_ksndmc_df['Date_Time'], inplace=True)
# weather_ksndmc_df.sort_index(inplace=True)
# cols = weather_ksndmc_df.columns.tolist()
# cols = cols[-1:] + cols[:-1]
# weather_ksndmc_df = weather_ksndmc_df[cols]
# # print weather_ksndmc_df.head()
# print("Mean values - Humidty, Temp, wind speed")
# print weather_ksndmc_df['HUMIDITY'].mean()
# print weather_ksndmc_df['TEMPERATURE'].mean()
# print weather_ksndmc_df['WIND_SPEED'].mean()
# print("Min values - Humidty, Temp, wind speed")
# print weather_ksndmc_df['HUMIDITY'].min()
# print weather_ksndmc_df['TEMPERATURE'].min()
# print weather_ksndmc_df['WIND_SPEED'].min()
# print("Max values - Humidty, Temp, wind speed")
# print weather_ksndmc_df['HUMIDITY'].max()
# print weather_ksndmc_df['TEMPERATURE'].max()
# print weather_ksndmc_df['WIND_SPEED'].max()
# print "smallest and largest 3, Hum, temp, wind speed"
# # print weather_ksndmc_df.nsmallest(5, ['HUMIDITY','TEMPERATURE', 'WIND_SPEED'], keep='last')
# print weather_ksndmc_df['2014-06-15 10:30:00':'2014-06-15 22:30:00']
# raise SystemExit(0)



