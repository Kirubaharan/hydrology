__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
aral_rain_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/ksndmc_rain.csv'
had_rain_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/ksndmc_rain.csv'

aral_df = pd.read_csv(aral_rain_file, sep=',', header=0)
had_df = pd.read_csv(had_rain_file, sep=',', header=0)
datetime_format = '%Y-%m-%d %H:%M:%S'
aral_df['Date_Time'] = pd.to_datetime(aral_df['Date_Time'], format=datetime_format)
aral_df.set_index(aral_df['Date_Time'], inplace=True)
aral_df.sort_index(inplace=True)
aral_df = aral_df.drop('Date_Time', 1)
aral_w_df = aral_df.resample('W-MON', how=np.sum)
aral_w_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/aral_weekly_rain.csv')
had_df['Date_Time'] = pd.to_datetime(had_df['Date_Time'], format=datetime_format)
had_df.set_index(had_df['Date_Time'], inplace=True)
had_df.sort_index(inplace=True)
had_df = had_df.drop('Date_Time', 1)
had_w_df = had_df.resample('W-MON', how=np.sum)
had_w_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/had_weekly_rain.csv')
print aral_df.head()
print had_df.head()