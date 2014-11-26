__author__ = 'kiruba'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from datetime import timedelta
import itertools

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)


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
    # first_time = pd.to_datetime('2014-05-15 18:00:00', format='%Y-%m-%d %H:%M:%S')
    # final_time = pd.to_datetime('2014-09-09 23:00:00', format='%Y-%m-%d %H:%M:%S')
    for key, value in param.items():
        # print key
        # print len(wrong_date_time)
        if value[1] == '>':
            wrong_df = dataframe[dataframe[key] > value[0]]
        if value[1] == '<':
            wrong_df = dataframe[dataframe[key] < value[0]]
        if value[1] ==  '=':
            wrong_df = dataframe[dataframe[key] == value[0]]
        for wrong_time in wrong_df.index:
            if max(dataframe.index) > wrong_time > min(dataframe.index):
                wrong_date_time.append(wrong_time)
            # if final_time > wrong_time > first_time:


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
    initial_cutoff = min(dataframe.index) + timedelta(days=1)
    final_cutoff = max(dataframe.index) - timedelta(days=1)
    for date_time in wrong_date_time:
        if (date_time > initial_cutoff ) and (date_time < final_cutoff):
            prev_date_time = date_time - timedelta(days=1)
            next_date_time = date_time + timedelta(days=1)
            prev_value = dataframe[column_name][prev_date_time.strftime('%Y-%m-%d %H:%M')]
            next_value = dataframe[column_name][next_date_time.strftime('%Y-%m-%d %H:%M')]
            average_value = 0.5*(prev_value + next_value)
            dataframe[column_name][date_time.strftime('%Y-%m-%d %H:%M')] = average_value

    return dataframe


base_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/smgoll_08_05_11_25_2014.CSV'
#read csv file
df_base = pd.read_csv(base_file, header=0, sep='\t')
#Drop seconds
df_base['Time'] = df_base['Time'].map(lambda x: x[ :5])
# convert date and time columns into timestamp
# print df_base.head()
date_format = '%d/%m/%y %H:%M'
df_base['Date_Time'] = pd.to_datetime(df_base['Date'] + ' ' + df_base['Time'], format=date_format)
df_base.set_index(df_base['Date_Time'], inplace=True)
# df_base = dt_time(df_base)
# print df_base.index.hour
# print df_base.head()
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
print df_base.head()

"""
Remove Duplicates
"""
# # print df_base.count()

# print df_base.head()
# print df_base.count()

df_base['index'] = df_base.index
df_base.drop_duplicates(subset='index', take_last=True, inplace=True)
del df_base['index']
df_base = df_base.sort()
# df_base = df_base["2014-05-14 18:30":"2014-09-10 23:30"]
"""
Fill in missing values interpolate
"""

# print .head()
# print weather_df.tail()
# # print weather_df.count()
new_index = pd.date_range(start=min(df_base.index), end=max(df_base.index), freq='30min' )
# # print len(new_index)
# # print new_index
# # print df_base.index.get_duplicates()
df_base = df_base.reindex(index=new_index, method=None)
# # print df_base.count()
df_base = df_base.interpolate(method='time')
# print rain_df.head()
# remove unneccessary columns
weather_df = df_base.drop(['Date',
                           'Time',
                           "Date_Time",
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

# select values where ksndmc data is available

# print df_base["2014-06-30"]
#rain df
rain_df = df_base[['Rain Collection (mm)']]

#  raw data
# Max Air temperature
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_df.index, weather_df["Max Air Temperature (C)"], 'r-', label='Air Temperature (C)')
plt.ylabel(r'\textbf{Temperature}($^\circ$C)')
plt.title(r"Maximum Temperature($^\circ$C) - Aralumallige", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/sm_max_temp')
fig.autofmt_xdate(rotation=90)

#Min Air Temperature
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_df.index, weather_df["Min Air Temperature (C)"], 'r-', label='Min Air Temperature (C)')
plt.ylabel(r'\textbf{Temperature}($^\circ$C)')
plt.title(r"Minimum Temperature($^\circ$C) - Aralumallige", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/sm_min_temp')
fig.autofmt_xdate(rotation=90)
#  wind speed
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_df.index, weather_df['Wind Speed (kmph)'], 'r-', label='Wind Speed (Kmph)')
plt.ylabel(r'\textbf{Wind Speed}($Km/h$)')
plt.title(r"Wind Speed - Aralumallige", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/sm_wind_speed')
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
col_cutoff_dict = {'Max Air Temperature (C)': [45, '>'],
                   'Min Air Temperature (C)': [0, '<'],
                    'Max Wind Speed (kmph)': [50, '>'],
                    'Wind Speed (kmph)': [0.0, '=']}

# weather_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/corrected_weather.csv')

wrong_timestamps = pick_incorrect_value(df_base, **col_cutoff_dict)
#plot weather parameters
# print weather_df.head()
for column_name in ['Max Air Temperature (C)', 'Min Air Temperature (C)', 'Max Wind Speed (kmph)', 'Wind Speed (kmph)']:
    day_interpolate(weather_df, column_name, wrong_timestamps)





weather_df.index.name = "Date_Time"
# print weather_df.head()
# print weather_df["2014-06-30"]
weather_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/corrected_weather.csv')

# print weather_df["2014-06-30"]



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
# print weather_df.head()

#min wind speed
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_df.index, weather_df['Min Wind Speed (kmph)'], 'g-', label='Wind Speed (Kmph)')
plt.ylabel(r'\textbf{Wind Speed}($Km/h$)')
plt.title(r"Calibrated Minimum Wind Speed - Aralumallige", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/corr_min_wind_speed_aral')
fig.autofmt_xdate(rotation=90)
# plt.show()

# max wind speed
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_df.index, weather_df['Max Wind Speed (kmph)'], 'g-', label='Wind Speed (Kmph)')
plt.ylabel(r'\textbf{Wind Speed}($Km/h$)')
plt.title(r"Maximum Wind Speed - Aralumallige", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/corr_max_wind_speed_aral')
fig.autofmt_xdate(rotation=90)
# plt.show()


# print wrong_timestamps
## Rainfall correction
ksndmc_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/kanaswadi/KSNDMC_01-05-2014_10-09-2014_KANASAWADI.csv'
ksndmc_df = pd.read_csv(ksndmc_file, sep=',')

# drop unnecessary
ksndmc_df.drop(['TRGCODE', 'DISTRICT', 'TALUKNAME', 'HOBLINAME', 'HOBLICODE', 'Total'], inplace=True, axis=1)
# print ksndmc_df.head()
# reshape the dataframe
data = []
for row_no, row in ksndmc_df.iterrows():
    # print row_no
    date = row['Date']
    # print row
    for time, value in row.ix[1:, ].iteritems():
        data.append((date, time, value))
        # print x
        # print y

# print data
# create date time index
data_df = pd.DataFrame(data, columns=['Date', 'Time', 'Rain(mm)'])
# print data_df.head()
date_format ="%d-%b-%Y %H:%M"
data_df['Date_Time'] = pd.to_datetime(data_df['Date'] + ' ' + data_df['Time'], format=date_format)
data_df.set_index(data_df['Date_Time'], inplace=True)
data_df.sort_index(inplace=True)
data_df.drop(['Date_Time', 'Date', 'Time'], axis=1, inplace=True)
# Perform cumulative difference
data_8h_df = data_df['2014-05-01 8H30T': '2014-09-10 8H30T']
data_8h_df['diff'] = 0.000
# print data_8h_df.head()


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2,s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

for d1, d2 in pairwise(data_8h_df.index):
    # print d1, d2
    if data_8h_df['Rain(mm)'][d2] > data_8h_df['Rain(mm)'][d1]:
        data_8h_df['diff'][d2] = data_8h_df['Rain(mm)'][d2] - data_8h_df['Rain(mm)'][d1]

data_8h_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/kanaswadi/recal_kanaswadi.csv')
# print data_8h_df.index.min(), data_8h_df.index.max(), data_8h_df.index.is_monotonic
data_30min_df = data_8h_df.resample('30Min', how=np.sum, label='right', closed='right')
data_30min_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/kanaswadi/reshaped_30_min_kanaswadi.csv')
# print data_8h_df['2014-05-08 01:30:00': '2014-05-08 02:30:00']
# print data_30min_df['2014-05-08 01:00:00': '2014-05-08 02:30:00']
# print np.sum(data_8h_df['diff'], axis=1)
# print np.sum(data_30min_df['diff'], axis=1)
# print wrong_timestamps
# print rain_df['2014-05-20 19:00:00':'2014-05-20 21:00:00']
initial_ksndmc_cutoff = min(data_30min_df.index)
final_ksndmc_cutoff = max(data_30min_df.index)
for wrong_datetime in wrong_timestamps:
    if (wrong_datetime >= initial_ksndmc_cutoff) and (wrong_datetime <= final_ksndmc_cutoff):
        a = wrong_datetime.strftime('%Y-%m-%d %H:%M')
        rain_df['Rain Collection (mm)'][a] = data_30min_df['diff'][a]
#
# print rain_df['2014-05-20 19:00:00':'2014-05-20 21:00:00']
# print rain_df['2014-05-08 01:00': '2014-05-08 02:30']
print rain_df['2014-06-30']
rain_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/corrected_rain.csv')
# print rain_df.head()
rain_df = rain_df.resample('3H', how=np.sum, label='right', closed='right')
# Plot
#check dam caliberation
y_cal = [10, 40, 100, 160, 225, 275, 300]
x_cal = [2036, 2458, 3025, 4078, 5156, 5874, 6198]


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
##stage_calibration
a_stage = polyfit(x_cal, y_cal, 1)
# po_stage = np.polyfit(x_cal, y_cal, 1)
# f_stage = np.poly1d(po_stage)
# print np.poly1d(f_stage)
# print a_stage
# print a_stage['polynomial'][0]
coeff_cal = a_stage['polynomial']

x_cal_new = np.linspace(min(x_cal), max(x_cal), 50)
y_cal_new = (x_cal_new*coeff_cal[0]) + coeff_cal[1]

fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(x_cal, y_cal, 'bo', label=r'Observation')
plt.plot(x_cal_new, y_cal_new, 'b-', label=r'Prediction')
plt.xlim([(min(x_cal)-500), (max(x_cal)+500)])
plt.ylim([0, 350])
plt.xlabel(r'\textbf{Capacitance} (ohm)')
plt.ylabel(r'\textbf{Stage} (m)')
plt.legend(loc='upper left')
plt.title(r'Capacitance Sensor Calibration for 591 Check dam')
plt.text(x=1765, y=275, fontsize=15, s=r"\textbf{{$ y = {0:.1f} x  {1:.1f} $}}".format(coeff_cal[0], coeff_cal[1]))
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/sensor_calib_591')
# plt.show()

## Read check dam data
block_1 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525_008_001.CSV'
water_level_1 = pd.read_csv(block_1, skiprows=9, sep=',', header=0,  names=['scan no', 'date',
                                                                            'time', 'raw value', 'calibrated value'])
water_level_1['calibrated value'] = (water_level_1['raw value']*coeff_cal[0]) + coeff_cal[1]  # in cm
# convert to metre
water_level_1['calibrated value'] /= 100
#change the column name
water_level_1.columns.values[4] = 'stage(m)'
# create date time index
format = '%d/%m/%Y  %H:%M:%S'
# change 24:00:00 to 23:59:59
water_level_1['time'] = water_level_1['time'].replace(' 24:00:00', ' 23:59:59')
water_level_1['date_time'] = pd.to_datetime(water_level_1['date'] + water_level_1['time'], format=format)
water_level_1.set_index(water_level_1['date_time'], inplace=True)
# drop unneccessary columns before datetime aggregation
water_level_1.drop(['scan no', 'date', 'time', 'raw value'], inplace=True, axis=1)
#aggregate daily
water_level_1 = water_level_1.resample('3H', how=np.mean)
# print water_level_1
block_2 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525_008_002.CSV'
water_level_2 = pd.read_csv(block_2, skiprows=9, sep=',', header=0,  names=['scan no', 'date', 'time', 'raw value', 'calibrated value'])
water_level_2['calibrated value'] = (water_level_2['raw value']*coeff_cal[0]) + coeff_cal[1] # in cm
# convert to metre
water_level_2['calibrated value'] /= 100
#change the column name
water_level_2.columns.values[4] = 'stage(m)'
# create date time index
format = '%d/%m/%Y  %H:%M:%S'
# change 24:00:00 to 23:59:59
water_level_2['time'] = water_level_2['time'].replace(' 24:00:00', ' 23:59:59')
water_level_2['date_time'] = pd.to_datetime(water_level_2['date'] + water_level_2['time'], format=format)
water_level_2.set_index(water_level_2['date_time'], inplace=True)
# drop unneccessary columns before datetime aggregation
water_level_2.drop(['scan no', 'date', 'time', 'raw value'], inplace=True, axis=1)
#aggregate daily
water_level_2 = water_level_2.resample('3H', how=np.mean)
# print water_level_2
block_3 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525_008_003.CSV'
water_level_3 = pd.read_csv(block_3, skiprows=9, sep=',', header=0,  names=['scan no', 'date', 'time', 'raw value', 'calibrated value'])
water_level_3['calibrated value'] = (water_level_3['raw value']*coeff_cal[0]) + coeff_cal[1] # in cm
# convert to metre
water_level_3['calibrated value'] /= 100
#change the column name
water_level_3.columns.values[4] = 'stage(m)'
# create date time index
format = '%d/%m/%Y  %H:%M:%S'
# change 24:00:00 to 23:59:59
water_level_3['time'] = water_level_3['time'].replace(' 24:00:00', ' 23:59:59')
water_level_3['date_time'] = pd.to_datetime(water_level_3['date'] + water_level_3['time'], format=format)
water_level_3.set_index(water_level_3['date_time'], inplace=True)
# drop unneccessary columns before datetime aggregation
water_level_3.drop(['scan no', 'date', 'time', 'raw value'], inplace=True, axis=1)
#aggregate daily
water_level_3 = water_level_3.resample('3H', how=np.mean)
# print water_level_3
water_level = pd.concat([water_level_1, water_level_2, water_level_3], axis=0)
# print water_level
# weather_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/corrected_weather.csv')
# rain_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/corrected_rain_data.csv')

# select rain data where stage is available
# rain_df = rain_df[min(water_level.index): max(water_level.index)]

# plot 3 hourly

fig, ax1 = plt.subplots(figsize=(11.69, 8.27))
ax1.plot(rain_df.index, rain_df["Rain Collection (mm)"], '-b', label='Rain(mm)')
# plt.plot([0, 1.9], [1, 1.9], '-k')
ax1.legend(loc='upper left')
for t1 in ax1.get_yticklabels():
    t1.set_color('b')
ax2 = ax1.twinx()
ax2.plot(water_level.index, water_level['stage(m)'], 'r-', label='Stage (m)')
ax2.hlines(1.9, min(water_level.index), max(water_level.index))
# plt.ylabel(r'\textbf{Wind Speed}($Km/h$)')
for t1 in ax2.get_yticklabels():
    t1.set_color('r')
plt.title(r"Average 3 Hourly Water Level in Checkdam 591", fontsize=20)
plt.legend(loc='upper right')
fig.autofmt_xdate(rotation=90)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/three_corr_hour_stage_591')

fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(rain_df.index, rain_df["Rain Collection (mm)"], '-b')
fig.autofmt_xdate(rotation=90)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/rainfall_corr_3_H_591')
# plt.show()

# weather_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/corrected_weather.csv')
# rain_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/corrected_rain_data.csv')