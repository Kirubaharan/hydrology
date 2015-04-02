__author__ = 'kiruba'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import operator


base_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/smgoll_5_8_14.csv'
#read csv file
df_base = pd.read_csv(base_file, header=0, sep='\t')
# convert date and time columns into timestamp
date_format = '%d/%m/%y %H:%M:%S'
df_base['Date_Time'] = pd.to_datetime(df_base['Date'] + ' ' + df_base['Time'], format=date_format)
# set index
df_base.set_index(df_base['Date_Time'], inplace=True)
print df_base.head()
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
# print df_base.head()
# print df_base.columns.values[16]
ksndmc_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/kanaswadi/KSNDMC_01-05-2014_10-09-2014_KANASAWADI.csv'
ksndmc_df = pd.read_csv(ksndmc_file, sep=',')

# drop unnecessary
ksndmc_df.drop(['TRGCODE', 'DISTRICT', 'TALUKNAME', 'HOBLINAME', 'HOBLICODE', 'Total'], inplace=True, axis=1)
# print ksndmc_df.head()
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
data_df = pd.DataFrame(data, columns=['Date', 'Time', 'Rain(mm)'])
data_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/kanaswadi/reshaped_kanaswadi.csv')
date_format ="%d-%b-%Y %H:%M"
data_df['Date_Time'] = pd.to_datetime(data_df['Date'] + ' ' + data_df['Time'], format=date_format)
data_df.set_index(data_df['Date_Time'], inplace=True)
data_df.sort_index(inplace=True)
data_df.drop(['Date_Time'], axis=1, inplace=True)
fig = plt.figure(figsize=(11.69, 8.27))
plt.title('Raw data')
plt.plot_date(data_df.index, data_df['Rain(mm)'], '-g')
fig.autofmt_xdate()
data_8h_df = data_df['2014-05-01 8H30T': '2014-09-10 8H30T']
# print data_8h_df.head()
# print data_df.head()


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2,s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

data_8h_df['diff'] = 0.000
# print data_8h_df
for d1, d2 in pairwise(data_8h_df.index):
    # print d1, d2
    if data_8h_df['Rain(mm)'][d2] > data_8h_df['Rain(mm)'][d1]:
        data_8h_df['diff'][d2] = data_8h_df['Rain(mm)'][d2] - data_8h_df['Rain(mm)'][d1]
#

# print data_8h_df
# data_df.plot(y='Rain(mm)', style='-b')
data_8h_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/kanaswadi/recal.csv')
# plt.show()
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(data_8h_df.index, data_8h_df['diff'], '-r')
fig.autofmt_xdate()
plt.show()
rain_k_df = data_8h_df[['diff']]
rain_k_df = rain_k_df.resample('3H', how=np.sum)
rain_k_df.columns.values[0] = 'Rain Collection (mm)'
# print rain_k_df
"""
Aggregate half hourly to daily
"""

# ## separate out rain daily sum
rain_df = df_base[['Date_Time', 'Rain Collection (mm)']]
rain_df = rain_df.resample('3H', how=np.sum)

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

# select rain data where stage is available
rain_df = rain_df[min(water_level.index): max(water_level.index)]

# plot 3 hourly

fig, ax1 = plt.subplots(figsize=(11.69, 8.27))
ax1.plot(rain_df.index, rain_df["Rain Collection (mm)"], '-b', label='Rain(mm)')
# plt.plot([0, 1.9], [1, 1.9], '-k')
ax1.legend()
ax2 = ax1.twinx()
ax2.plot(water_level.index, water_level['stage(m)'], 'r-', label='Stage (m)')
ax2.hlines(1.9, min(water_level.index), max(water_level.index))
# plt.ylabel(r'\textbf{Wind Speed}($Km/h$)')
plt.title(r"Average 3 Hourly Water Level in Checkdam 591", fontsize=16)
plt.legend(loc='upper left')
fig.autofmt_xdate(rotation=90)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/three_hour_stage_591')

fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(rain_df.index, rain_df["Rain Collection (mm)"], '-b')
fig.autofmt_xdate(rotation=90)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/rainfall_3_H_591')
plt.show()