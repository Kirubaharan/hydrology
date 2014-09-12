__author__ = 'kiruba'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from datetime import timedelta

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

base_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/smgoll_5_8_14.csv'
#read csv file
df_base = pd.read_csv(base_file, header=0, sep='\t')
# convert date and time columns into timestamp
date_format = '%d/%m/%y %H:%M:%S'
df_base['Date_Time'] = pd.to_datetime(df_base['Date'] + ' ' + df_base['Time'], format=date_format)
# set index
df_base.set_index(df_base['Date_Time'], inplace=True)
# sort based on index
df_base.sort_index(inplace=True)
cols = df_base.columns.tolist()
cols = cols[-1:] + cols[:-1]  # bring last column to first
df_base = df_base[cols]

## change column names that has degree symbols to avoid non-ascii error
df_base.columns.values[7] = 'Air Temperature (C)'
df_base.columns.values[8] = 'Min Air Temperature (C)'
df_base.columns.values[9] = 'Max Air Temperature (C)'
df_base.columns.values[16] = 'Canopy Temperature (C)'

# remove unneccessary columns
weather_df = df_base.drop(['Date',
                           'Time',
                           'Rain Collection (mm)',
                           'Barometric Pressure (KPa)',
                           'Soil Moisture',
                           'Leaf Wetness',
                           'Canopy Temperature (C)',
                           'Evapotranspiration',
                           'Charging status',
                           'Solar panel voltage',
                           'Network strength',
                           'Battery strength'], axis=1)



#plot weather parameters
# print weather_df.head()
#windspeed
# fig = plt.figure(figsize=(11.69, 8.27))
# plt.plot_date(weather_df.index, weather_df['Wind Speed (kmph)'], 'r-', label='Wind Speed (Kmph)')
# plt.ylabel(r'\textbf{Wind Speed}($Km/h$)')
# plt.title(r"Wind Speed - Aralumallige", fontsize=16)
# plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/wind_speed_aral')
# fig.autofmt_xdate(rotation=90)
#Min Air Temperature
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_df.index, weather_df["Min Air Temperature (C)"], 'r-', label='Min Air Temperature (C)')
plt.ylabel(r'\textbf{Temperature}($^\circ$C)')
plt.title(r"Minimum Temperature($^\circ$C) - Aralumallige", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/sm_min_temp')
fig.autofmt_xdate(rotation=90)

fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_df.index, weather_df["Max Air Temperature (C)"], 'r-', label='Air Temperature (C)')
plt.ylabel(r'\textbf{Temperature}($^\circ$C)')
plt.title(r"Maximum Temperature($^\circ$C) - Aralumallige", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/sm_max_temp')
fig.autofmt_xdate(rotation=90)

#humidity
# fig = plt.figure(figsize=(11.69, 8.27))
# plt.plot_date(weather_df.index, weather_df['Humidity (%)'], 'r-', label='Humidity (%)')
# plt.ylabel(r'\textbf{Humidity}(\%)')
# plt.title(r"Humidity - Aralumallige", fontsize=16)
# plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/humidity')
# fig.autofmt_xdate(rotation=90)
# #solar radiation
# fig = plt.figure(figsize=(11.69, 8.27))
# plt.plot_date(weather_df.index, weather_df['Solar Radiation (W/mm2)'], 'r-', label='Solar Radiation (W/mm2)')
# plt.ylabel(r'\textbf{Solar Radiation ($W/mm2$)')
# plt.title(r"Solar Radiation - Aralumallige", fontsize=16)
# plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/solar_rad')
# fig.autofmt_xdate(rotation=90)
# plt.show()


def weather_interpolate(dataframe, column_name, cutoff_value, mode):
    """
    Takes the average of previous time period and next time period
    :param dataframe: Weather pandas dataframe with datetime index
    :param column_name: Weather variable
    :type column_name: str
    :param cutoff_value: Cutoff value
    :type cutoff_value: float
    :param mode: 1 for Less than cutoff_value, 2 for more than cutoff value
    :type mode: int
    :return:corrected dataframe
    """
    # from datetime import timedelta
    if mode == 1:
        wrong_value = dataframe[dataframe[column_name] < cutoff_value]
    if mode == 2:
        wrong_value = dataframe[dataframe[column_name] > cutoff_value]
    for date_time in wrong_value.index:
        prev_date_time = date_time - timedelta(days=1)
        next_date_time = date_time + timedelta(days=1)
        prev_value = dataframe[column_name][prev_date_time.strftime('%Y-%m-%d %H:%M')]
        next_value = dataframe[column_name][next_date_time.strftime('%Y-%m-%d %H:%M')]
        average_value = (prev_value + next_value) / 2
        dataframe[column_name][date_time.strftime('%Y-%m-%d %H:%M')] = average_value
    return dataframe

#calibration
# wrong_min_temp = weather_df[weather_df['Min Air Temperature (C)'] < 0]
# # print wrong_temp.index
# for date_time in wrong_min_temp.index:
#     # print date_time
#     prev_date_time = date_time - timedelta(days=1)
#     prev_value = weather_df['Min Air Temperature (C)'][prev_date_time.strftime('%Y-%m-%d %H:%M')]
#     next_date_time = date_time + timedelta(days=1)
#     next_value = weather_df['Min Air Temperature (C)'][next_date_time.strftime('%Y-%m-%d %H:%M')]
#     average_value = (prev_value+next_value) / 2
#     weather_df['Min Air Temperature (C)'][date_time.strftime('%Y-%m-%d %H:%M')] = average_value

# weather_df = weather_interpolate(dataframe=weather_df, column_name='Min Air Temperature (C)',
#                                  cutoff_value=0, mode=1)

fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_df.index, weather_df["Min Air Temperature (C)"], 'r-', label='Min Air Temperature (C)')
plt.ylabel(r'\textbf{Temperature}($^\circ$C)')
plt.title(r"Calibrated Minimum Temperature($^\circ$C) - Aralumallige", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/sm_corr_min_temp')
fig.autofmt_xdate(rotation=90)
# plt.show()

# wrong_max_temp = weather_df[weather_df['Max Air Temperature (C)'] > 45]
# # print wrong_temp.index
# for date_time in wrong_max_temp.index:
#     print date_time
#     prev_date_time = date_time - timedelta(days=1)
#     prev_value = weather_df['Max Air Temperature (C)'][prev_date_time.strftime('%Y-%m-%d %H:%M')]
#     next_date_time = date_time + timedelta(days=1)
#     next_value = weather_df['Max Air Temperature (C)'][next_date_time.strftime('%Y-%m-%d %H:%M')]
#     average_value = (prev_value+next_value) / 2
#     weather_df['Max Air Temperature (C)'][date_time.strftime('%Y-%m-%d %H:%M')] = average_value
#



fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_df.index, weather_df["Max Air Temperature (C)"], 'r-', label='Air Temperature (C)')
plt.ylabel(r'\textbf{Temperature}($^\circ$C)')
plt.title(r"Calibrated Maximum Temperature($^\circ$C) - Aralumallige", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/sm_corr_max_temp')
fig.autofmt_xdate(rotation=90)

# plt.show()
print weather_df.head()

#min wind speed
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_df.index, weather_df['Min Wind Speed (kmph)'], 'g-', label='Wind Speed (Kmph)')
plt.ylabel(r'\textbf{Wind Speed}($Km/h$)')
plt.title(r"Minimum Wind Speed - Aralumallige", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/min_wind_speed_aral')
fig.autofmt_xdate(rotation=90)
# plt.show()

# max wind speed
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_df.index, weather_df['Max Wind Speed (kmph)'], 'g-', label='Wind Speed (Kmph)')
plt.ylabel(r'\textbf{Wind Speed}($Km/h$)')
plt.title(r"Maximum Wind Speed - Aralumallige", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/max_wind_speed_aral')
fig.autofmt_xdate(rotation=90)
# plt.show()

col_cutoff_dict = {'Max Air Temperature (C)': [45, '>'],
                   'Min Air Temperature (C)': [0, '<'],
                    'Max Wind Speed (kmph)': [50, '>']}


def pick_incorrect_value(dataframe, **param):
    """
    Selects a unique list of timestamp that satisfies the condition given in the param dictionary
    :param dataframe: Pandas dataframe
    :param param: Conditonal Dictionary, Eg.{column name: [cutoff, '>']}
    :type param: dict
    :return: unique list of timestamp
    :rtype: list
    """
    wrong_date_time = []
    unique_list = []
    for key, value in param.items():
        # print key
        # print len(wrong_date_time)
        if value[1] == '>':
            wrong_df = dataframe[dataframe[key] > value[0]]
        if value[1] == '<':
            wrong_df = dataframe[dataframe[key] < value[0]]
        for wrong_time in wrong_df.index:
            wrong_date_time.append(wrong_time)

    for i in wrong_date_time:
        if i not in unique_list:
            unique_list.append(i)

    return unique_list


def day_interpolate(dataframe, column_name, wrong_date_time):
    """

    :param dataframe: Pandas dataframe
    :param column_name: Interpolation target column name of dataframe
    :type column_name: str
    :param wrong_date_time: List of error timestamp
    :type wrong_date_time: list
    :return: Corrected dataframe
    """
    for date_time in wrong_date_time:
        prev_date_time = date_time - timedelta(days=1)
        next_date_time = date_time + timedelta(days=1)
        prev_value = dataframe[column_name][prev_date_time.strftime('%Y-%m-%d %H:%M')]
        next_value = dataframe[column_name][next_date_time.strftime('%Y-%m-%d %H:%M')]
        average_value = (prev_value + next_value) / 2
        dataframe[column_name][date_time.strftime('%Y-%m-%d %H:%M')] = average_value

    return dataframe
