__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from spread import  spread
from scipy.optimize import curve_fit
import math
from matplotlib import rc
import email.utils as eutils
import time
import datetime
from datetime import timedelta

# latex parameters
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)

# Weather file
weather_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/corrected_weather.csv'
# Rain file
rain_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/corrected_rain.csv'

# convert to pandas dataframe
weather_df = pd.read_csv(weather_file, sep=',', header=0)
# set index
date_format = '%Y-%m-%d %H:%M:%S'
weather_df['Date_Time'] = pd.to_datetime(weather_df['Date_Time'], format=date_format)
weather_df.set_index(weather_df['Date_Time'], inplace=True)
# sort based on index
weather_df.sort_index(inplace=True)
# drop date time column
weather_df = weather_df.drop('Date_Time', 1)
# print weather_df.head()
# Rain data frame
rain_df = pd.read_csv(rain_file, sep=',', header=0)
# set index

rain_df['Date_Time'] = pd.to_datetime(rain_df['Date_Time'], format=date_format)
rain_df.set_index(rain_df['Date_Time'], inplace=True)
# sort based on index
rain_df.sort_index(inplace=True)
# drop date time column
rain_df = rain_df.drop('Date_Time', 1)

# print rain_df.head()

"""
Check dam calibration
"""
# Polynomial fitting function


def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    results['polynomial'] = coeffs.tolist()
    #r squared
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y-ybar)**2)
    results['determination'] = ssreg/sstot
    return results

#check dam calibration values
y_cal = [10, 40, 100, 160, 225, 275, 300]
x_cal = [2036, 2458, 3025, 4078, 5156, 5874, 6198]
a_stage = polyfit(x_cal, y_cal, 1)
# coefficients of polynomial are stored in following list
coeff_cal = a_stage['polynomial']

# water_level_1 = pd.read_csv(block_1, skiprows=9, sep=',', header=0,  names=['scan no', 'date',
#                                                                             'time', 'raw value', 'calibrated value'])
# water_level_1['calibrated value'] = (water_level_1['raw value']*coeff_cal[0]) + coeff_cal[1]  # in cm
# # convert to metre
# water_level_1['calibrated value'] /= 100
# #change the column name
# water_level_1.columns.values[4] = 'stage(m)'
# # create date time index
# format = '%d/%m/%Y  %H:%M:%S'
# # change 24:00:00 15 May to 00:00:00 16 May
#
# # water_level_1['time'] = water_level_1['time'].replace(' 24:00:00', ' 23:59:59')
#
# # water_level_1['date'] = pd.to_datetime(water_level_1['date'], format='%d/%m/%Y ')
# # print water_level_1.head(20)
# # c_str = water_level_1['time'][11]
# c_str = ' 24:00:00'
# # print(c_str)
# for index, row in water_level_1.iterrows():
#     x_str = row['time']
#     # store stings in object and then compare
#     if x_str == c_str:
#         # print row['date']
#         # x_date = row['date']
#         # print x_date
#         # convert string to datetime object
#         r_date = pd.to_datetime(row['date'], format='%d/%m/%Y ')
#         # print r_date
#         # add 1 day
#         c_date = r_date + timedelta(days=1)
#         # convert datetime to string
#         c_date = c_date.strftime('%d/%m/%Y ')
#         # print(c_date)
#         # print row['date']
#         c_time = ' 00:00:00'
#         water_level_1['date'][index] = c_date
#         water_level_1['time'][index] = c_time
#
# water_level_1['date_time'] = pd.to_datetime(water_level_1['date'] + water_level_1['time'], format=format)
# water_level_1.set_index(water_level_1['date_time'], inplace=True)
# # # drop unneccessary columns before datetime aggregation
# water_level_1.drop(['scan no', 'date', 'time', 'raw value', 'date_time'], inplace=True, axis=1)
#
# print water_level_1.head(20)


def read_correct_ch_dam_data(csv_file):
    """
    Function to read, calibrate and convert time format (day1 24:00:00
    to day 2 00:00:00) in check dam data
    :param csv_file:
    :return: calibrated and time corrected data
    """
    water_level = pd.read_csv(csv_file, skiprows=9, sep=',', header=0, names=['scan no', 'date', 'time', 'raw value', 'calibrated value'])
    water_level['calibrated value'] = (water_level['raw value'] *coeff_cal[0]) + coeff_cal[1] #in cm
    water_level['calibrated value'] /= 100  #convert to metre
    # #change the column name
    water_level.columns.values[4] = 'stage(m)'
    # create date time index
    format = '%d/%m/%Y  %H:%M:%S'
    c_str = ' 24:00:00'
    for index, row in water_level.iterrows():
        x_str = row['time']
        if x_str == c_str:
            # convert string to datetime object
            r_date = pd.to_datetime(row['date'], format='%d/%m/%Y ')
            # add 1 day
            c_date = r_date + timedelta(days=1)
            # convert datetime to string
            c_date = c_date.strftime('%d/%m/%Y ')
            c_time = ' 00:00:00'
            water_level['date'][index] = c_date
            water_level['time'][index] = c_time

    water_level['date_time'] = pd.to_datetime(water_level['date'] + water_level['time'], format=format)
    water_level.set_index(water_level['date_time'], inplace=True)
    # # drop unneccessary columns before datetime aggregation
    water_level.drop(['scan no', 'date', 'time', 'raw value', 'date_time'], inplace=True, axis=1)

    return water_level


## Read check dam data
block_1 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525_008_001.CSV'
water_level_1 = read_correct_ch_dam_data(block_1)
# print water_level_1.head(20)
block_2 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525_008_002.CSV'
water_level_2 = read_correct_ch_dam_data(block_2)
block_3 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525_008_003.CSV'
water_level_3 = read_correct_ch_dam_data(block_3)
block_4 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525_008_004.CSV'
water_level_4 = read_correct_ch_dam_data(block_4)
water_level = pd.concat([water_level_1, water_level_2, water_level_3, water_level_4], axis=0)
# print water_level.head(20)

"""
Join weather and rain data
"""
weather_df = weather_df.join(rain_df, how='right')
weather_df = weather_df[min(water_level.index): max(water_level.index)]
weather_df = weather_df.join(water_level, how='right')
# print weather_df.head(20)
