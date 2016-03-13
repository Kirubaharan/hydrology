__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import checkdam.meteolib as meteo


def datesep(df):
    """

    :param df: dataframe
    :param column_name: date column name
    :return: date array, month array, year array
    """

    date = pd.DatetimeIndex(df.index).day
    month = pd.DatetimeIndex(df.index).month
    year = pd.DatetimeIndex(df.index).year
    return date, month, year


date_format = '%Y-%m-%d %H:%M:%S'
daily_format = '%Y-%m-%d'

weather_file = '/media/kiruba/New Volume/milli_watershed/tmg_lake_water_balance/tmg_open_water_evap.csv'
weather_df = pd.read_csv(weather_file, sep=',')
weather_df.drop('Unnamed: 0', 1, inplace=True)
weather_df['date_time'] = pd.to_datetime(weather_df['date_time'], format=daily_format)
weather_df.set_index(weather_df['date_time'], inplace=True)
weather_df.drop('date_time', 1, inplace=True)
print weather_df.head()
print(weather_df.tail())
# Rain file
rain_file = '/media/kiruba/New Volume/milli_watershed/cumulative impacts/had_rainfall_daily.csv'
# Rain data frame
rain_df = pd.read_csv(rain_file, sep=',', header=0)
# set index
rain_df['date_time'] = pd.to_datetime(rain_df['date_time'], format=daily_format)
rain_df.set_index(rain_df['date_time'], inplace=True)
# sort based on index
rain_df.sort_index(inplace=True)
rain_df.loc[:, 'rain(mm)'] = rain_df.loc[:, 'diff']
# drop date time column
rain_df.drop(['date_time', 'diff'], 1, inplace=True)

weather_rain_df = weather_df.join(rain_df, how='outer')

date, month, year = datesep(weather_rain_df)
weather_rain_df.loc[:, 'date'] = date
weather_rain_df.loc[:, 'month'] = month
weather_rain_df.loc[:, 'year'] = year
weather_rain_df.loc[:, 'doy'] = meteo.date2doy(dd=weather_rain_df.date, mm=weather_rain_df.month, yyyy=weather_rain_df.year)
# impute by mean
f = lambda x: x.fillna(x.mean())
doy_month_grouped = weather_rain_df.groupby(['month', 'doy'])
print(weather_rain_df.head())
