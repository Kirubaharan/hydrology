__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import checkdam.checkdam as cd
from matplotlib import rc
import matplotlib.cm as cmx
import matplotlib.colors as colors
from datetime import timedelta
import math
import ccy_classic_lstsqr

# latex parameters
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)

"""
Variables
"""
full_stage = 0.61  # check dam height, above this it is assumed check dam will overflow
date_format = '%Y-%m-%d %H:%M:%S'
daily_format = '%Y-%m-%d'
resolution_ody = 0.0008
stage_cutoff = 0

# ------------------------------------------------------------------#
# Weather file
weather_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/corrected_weather_ws.csv'
# Rain file
rain_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/ksndmc_rain.csv'
# convert to pandas dataframe
weather_df = pd.read_csv(weather_file, sep=',', header=0)
weather_df['Date_Time'] = pd.to_datetime(weather_df['Date_Time'], format=date_format)
weather_df.set_index(weather_df['Date_Time'], inplace=True)
weather_df.sort_index(inplace=True)
weather_df = weather_df.drop('Date_Time', 1)
# print weather_df.head()
# raise SystemExit(0)
# Rain data frame
rain_df = pd.read_csv(rain_file, sep=',', header=0)
# set index
rain_df['Date_Time'] = pd.to_datetime(rain_df['Date_Time'], format=date_format)
rain_df.set_index(rain_df['Date_Time'], inplace=True)
# sort based on index
rain_df.sort_index(inplace=True)
# drop date time column
rain_df = rain_df.drop('Date_Time', 1)

# raise SystemExit(0)
"""
Check dam calibration
"""
y_cal = np.array([100, 500, 1000, 2000, 2500, 3000])
x_cal = np.array([2018, 2761, 3492, 4924, 5609, 6320])
a_stage = cd.polyfit(x_cal, y_cal, 1)
coeff_cal = a_stage['polynomial']
slope = coeff_cal[0]
intercept = coeff_cal[1]

"""
Read Check dam data
"""
block_1 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2510/2510_006_001.CSV'
water_level_1 = cd.read_correct_ch_dam_data(block_1, slope, intercept, stage_cutoff=stage_cutoff)
block_2 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2510/2510_006_002_25_8_14.CSV'
water_level_2 = cd.read_correct_ch_dam_data(block_2, slope, intercept, stage_cutoff=stage_cutoff)
block_3 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2510/2510_006_003.CSV'
water_level_3 = cd.read_correct_ch_dam_data(block_3, slope, intercept, stage_cutoff=stage_cutoff)
block_4 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2510/2510_006_004.CSV'
water_level_4 = cd.read_correct_ch_dam_data(block_4, slope, intercept, stage_cutoff=stage_cutoff)
block_5 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2510/2510_002_001.CSV'
water_level_5 = cd.read_correct_ch_dam_data(block_5, slope, intercept, stage_cutoff=stage_cutoff)
block_6 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2510/2510_002_002_03_12_2014.CSV'
water_level_6 = cd.read_correct_ch_dam_data(block_6, slope, intercept, stage_cutoff=stage_cutoff)
block_7 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2510/2510_002_003_11_12_2014.CSV'
water_level_7 = cd.read_correct_ch_dam_data(block_7, slope, intercept, stage_cutoff=stage_cutoff)
block_8 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2510/2510_002_004_23_12_2014.CSV'
water_level_8 = cd.read_correct_ch_dam_data(block_8, slope, intercept, stage_cutoff=stage_cutoff)
block_9 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2510/2510_002_005_3_1_2015.CSV'
water_level_9 = cd.read_correct_ch_dam_data(block_9, slope, intercept, stage_cutoff=stage_cutoff)
for i in range(1, 10, 1):
    eval("water_level_{0}.drop(water_level_{0}.tail(1).index, inplace=True, axis=0)".format(i))
    eval("water_level_{0}.drop(water_level_{0}.head(1).index, inplace=True, axis=0)".format(i))
#
# for i in range(1, 10, 1):
#     print "water_level_{0}".format(i)
#     print eval("water_level_{0}.head()".format(i))

fig = plt.figure()
for i in range(1, 10, 1):
    x = eval("water_level_{0}.index".format(i))
    y = eval("water_level_{0}['stage(m)']".format(i))
    plt.plot(x, y)
#
plt.show()
# water_level = pd.concat([water_level_1, water_level_2, water_level_3, water_level_4, water_level_5, water_level_6, water_level_7, water_level_8, water_level_9], axis=0)
# water_level.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/stage_634.csv')
# raise SystemExit(0)
water_level_30min = pd.concat([water_level_1, water_level_2, water_level_3, water_level_4], axis=0)
water_level_30 = water_level_30min.sort()
rounded = np.array(water_level_30min.index, dtype='datetime64[m]')
water_level_30min = water_level_30min.set_index(rounded)
start_time_30 = min(water_level_30min.index)
end_time_30 = max(water_level_30min.index)
# new_index_30min = pd.date_range(start=start_time_30.strftime('%Y-%m-%d %H:%M'), end=end_time_30.strftime('%Y-%m-%d %H:%M'), freq='30min')
new_index_30 = pd.date_range(start=start_time_30, end=end_time_30, freq='30min')
water_level_30min = water_level_30min.reindex(new_index_30, method=None)
water_level_30min = water_level_30min.interpolate(method='time')
# water_level_30min = water_level_30min.set_index(new_index_30min)
water_level_30min.index.name = 'Date'
# print water_level_30min.head()
# raise SystemExit(0)
water_level_10min = pd.concat([water_level_5, water_level_6, water_level_7, water_level_8, water_level_9], axis=0)
water_level_10 = water_level_10min.sort()
rounded = np.array(water_level_10min.index, dtype='datetime64[m]')
water_level_10min = water_level_10min.set_index(rounded)
start_time_10 = min(water_level_10min.index)
end_time_10 = max(water_level_10min.index)
# new_index_10min = pd.date_range(start=start_time_10.strftime('%Y-%m-%d %H:%M'), end=end_time_10.strftime('%Y-%m-%d %H:%M'), freq='10min')
new_index_10 = pd.date_range(start=start_time_10, end=end_time_10, freq='10min')
water_level_10min = water_level_10min.reindex(new_index_10, method=None)
water_level_10min = water_level_10min.interpolate(method='time')
# water_level_10min = water_level_10min.set_index(new_index_10min)
water_level_10min.index.name = 'Date'

water_level = pd.concat([water_level_30min, water_level_10min], axis=0)
water_level = water_level.resample('30min', how=np.mean, label='right', closed='right')
# print water_level.head()
# water_level = water_level['2014-06-01':'2014-06-04']
# fig = plt.figure()
# plt.plot(water_level.index, water_level['stage(m)'], '-bo')
# plt.hlines(y=stage_cutoff, xmin=min(water_level.index), xmax=max(water_level.index))
# plt.hlines(y=full_stage, xmin=min(water_level.index), xmax=max(water_level.index), colors='g')
# plt.show()
# raise SystemExit(0)
# water_level['stage(m)'] = cd.myround(a=water_level['stage(m)'], decimals=2)
# print water_level.head()
water_level.loc[water_level['stage(m)'] < stage_cutoff, 'stage(m)'] = 0.0
# water_level = water_level[:"2014-12-08"]
# water_level[water_level['stage(m)'] < stage_cutoff] = 0
fig = plt.figure()
plt.plot(water_level.index, water_level['stage(m)'], 'r-')
plt.show()
raise SystemExit(0)
"""
Join weather and rain data
"""
weather_df = weather_df.join(rain_df, how='right')
weather_df = weather_df[min(water_level.index).strftime(daily_format): max(water_level.index).strftime(daily_format)]
weather_df = weather_df.join(water_level, how='inner')

"""
Remove Duplicates
"""
weather_df['index'] = weather_df.index
weather_df.drop_duplicates(subset='index', take_last=True, inplace=True)
del weather_df['index']
weather_df = weather_df.sort()
# print weather_df.head()
"""
Open water evaporation
"""
z = 838
p = (1 - (2.25577 * (10 ** -5) * z))
air_p_pa = 101325 * (p ** 5.25588)
# give air pressure value
weather_df['AirPr(Pa)'] = air_p_pa
"""
Half hourly Extraterrestrial Radiation Calculation(J/m2/30min)
"""
sc_default = 1367.0  # Solar constant in W/m^2 is 1367.0.
ch_634_lat = 13.365621
ch_634_long = 77.559051
weather_df['Rext (MJ/m2/30min)'] = 0.000
for i in weather_df.index:
    weather_df.loc[i.strftime('%Y-%m-%d %H:%M:%S'),'Rext (MJ/m2/30min)'] = (cd.extraterrestrial_irrad(local_datetime=i,
                                                                                                   latitude_deg=ch_634_lat,
                                                                                                   longitude_deg=ch_634_long))
"""
wind speed from km/h to m/s
1 kmph = 0.277778 m/s
"""
# weather_df['Wind Speed (mps)'] = weather_df['Wind Speed (kmph)'] * 0.277778
"""
Radiation unit conversion
"""
weather_df['Solar Radiation (MJ/m2/30min)'] = (weather_df['Solar Radiation (Wpm2)'] * 1800) / (10 ** 6)
"""
Average Temperature Calculation
"""
# weather_df['Average Temp (C)'] = 0.5 * (weather_df['Min Air Temperature (C)'] + weather_df['Max Air Temperature (C)'])

"""
Half hourly Evaporation calculation
"""
airtemp = weather_df['Air Temperature (C)']
hum = weather_df['Humidity (%)']
airpress = weather_df['AirPr(Pa)']
rs = weather_df['Solar Radiation (MJ/m2/30min)']
rext = weather_df['Rext (MJ/m2/30min)']
wind_speed = weather_df['Wind Speed (mps)']
weather_df['Evaporation (mm/30min)'] = cd.half_hour_evaporation(airtemp=airtemp, rh=hum, airpress=airpress,
                                                                rs=rs, rext=rext, u=wind_speed, z=z)


# raise SystemExit(0)
"""
Select data where stage is available
"""
weather_stage_avl_df = weather_df[min(water_level.index):max(water_level.index)]
"""
Convert observed stage to volume by linear interpolation
"""
# set stage as index
stage_vol_df = pd.read_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/stage_vol.csv',
                           sep=',', header=0, names=['sno', 'stage_m', 'total_vol_cu_m'])
# print stage_vol_df

stage_vol_df.drop('sno', inplace=True, axis=1)
stage_vol_df.set_index(stage_vol_df['stage_m'], inplace=True)
water_balance_df = weather_stage_avl_df[['rain (mm)', 'Evaporation (mm/30min)', 'stage(m)']]
water_balance_df['volume (cu.m)'] = 0.000
for index, row in water_balance_df.iterrows():
    obs_stage = row['stage(m)']  # observed stage
    if obs_stage >= stage_cutoff:
        x1, x2 = cd.find_range(stage_vol_df['stage_m'].tolist(), obs_stage)
        x_diff = x2 - x1
        y1 = stage_vol_df['total_vol_cu_m'][x1]
        y2 = stage_vol_df['total_vol_cu_m'][x2]
        y_diff = y2 - y1
        slope = y_diff / x_diff
        y_intercept = y2 - (slope * x2)
        water_balance_df.loc[index.strftime('%Y-%m-%d %H:%M:%S'), 'volume (cu.m)'] = (slope * obs_stage) + y_intercept

"""
full volume calculation
"""
x1, x2 = cd.find_range(stage_vol_df['stage_m'].tolist(), full_stage)
x_diff = x2 - x1
y1 = stage_vol_df['total_vol_cu_m'][x1]
y2 = stage_vol_df['total_vol_cu_m'][x2]
y_diff = y2 - y1
slope = y_diff / x_diff
y_intercept = y2 - (slope * x2)
full_volume = (slope*full_stage) + y_intercept
print full_volume

"""
Overflow
"""
# length_check_dam = 12.0
# width_check_dam = 0.6
# no_of_contractions = 0
water_balance_df['overflow(cu.m)'] = 0.000
for index, row in water_balance_df.iterrows():
    obs_volume = row['volume (cu.m)']
    if obs_volume > full_volume:
        overflow_volume = obs_volume - full_volume
        water_balance_df.loc[index.strftime('%Y-%m-%d %H:%M:%S'), 'overflow(cu.m)'] = overflow_volume

print water_balance_df.head()
# for index, row in water_balance_df.iterrows():
#     obs_stage = row['stage(m)']
#     if cd.myround(a=obs_stage, decimals=2) > full_stage:
#         effective_head = obs_stage - full_stage
#         previous_time = index - timedelta(seconds=1800)
#         if cd.myround(a=water_balance_df['stage(m)'][previous_time.strftime(date_format)], decimals=3) > full_stage:
#             water_balance_df['overflow(cu.m)'][index.strftime(date_format)] = 1800 * 1.84 * width_check_dam * (
#                 effective_head ** 1.5)
#         else:
#             x1 = 0
#             x2 = 1800
#             y1 = cd.myround(a=water_balance_df['stage(m)'][previous_time.strftime(date_format)], decimals=3)
#             y2 = cd.myround(a=obs_stage, decimals=3)
#             slope = (y1 - y2) / (x1 - x2)
#             intercept = y2 - (slope * x2)
#             time_of_overflow = 1800 - ((full_stage - intercept) / slope)
#             water_balance_df['overflow(cu.m)'][
#                 index.strftime(date_format)] = time_of_overflow * 1.84 * width_check_dam * (effective_head ** 1.5)
#
# water_balance_df = water_balance_df["2014-05-15":]
print "overflow"

"""
Stage vs area linear relationship
"""
stage_area_df = pd.read_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/cont_area.csv',
                            sep=',', header=0, names=['sno', 'stage_m', 'total_area_sq_m'])
stage_area_df.drop('sno', inplace=True, axis=1)
# set stage as index
stage_area_df.set_index(stage_area_df['stage_m'], inplace=True)
# create empty column
water_balance_df['ws_area(sq.m)'] = 0.000
for index, row in water_balance_df.iterrows():
    obs_stage = row['stage(m)']  # observed stage
    if obs_stage >= stage_cutoff:
        x1, x2 = cd.find_range(stage_area_df['stage_m'].tolist(), obs_stage)
        x_diff = x2 - x1
        y1 = stage_area_df['total_area_sq_m'][x1]
        y2 = stage_area_df['total_area_sq_m'][x2]
        y_diff = y2 - y1
        slope = y_diff / x_diff
        y_intercept = y2 - (slope * x2)
        water_balance_df.loc[index.strftime('%Y-%m-%d %H:%M:%S'), 'ws_area(sq.m)'] = (slope * obs_stage) + y_intercept
"""
Surface area to volume ratio
"""
average_stage = full_stage
x1, x2 = cd.find_range(stage_vol_df['stage_m'].tolist(), average_stage)
x_diff = x2 - x1
y1 = stage_vol_df['total_vol_cu_m'][x1]
y2 = stage_vol_df['total_vol_cu_m'][x2]
y_diff = y2 - y1
slope = y_diff / x_diff
y_intercept = y2 - (slope * x2)
average_volume = (slope*average_stage) + y_intercept
print average_stage
print average_volume
x1, x2 = cd.find_range(stage_area_df['stage_m'].tolist(), average_stage)
x_diff = x2 - x1
y1 = stage_area_df['total_area_sq_m'][x1]
y2 = stage_area_df['total_area_sq_m'][x2]
y_diff = y2 - y1
slope = y_diff / x_diff
y_intercept = y2 - (slope * x2)
average_area = (slope*average_stage) + y_intercept
print average_area
surface_area_to_vol_ratio = average_area/average_volume
print "surface area to vol ratio is %0.2f" %surface_area_to_vol_ratio
# raise SystemExit(0)
"""
Evaporation Volume estimation
"""
water_balance_df['Evaporation (cu.m)'] = (water_balance_df['Evaporation (mm/30min)'] * 0.001) * water_balance_df[
    'ws_area(sq.m)']
"""
Daily Totals of Rain, Evaporation, Overflow
"""
sum_df = water_balance_df[['rain (mm)', 'Evaporation (cu.m)', 'Evaporation (mm/30min)', 'overflow(cu.m)']]
sum_df = sum_df.resample('D', how=np.sum)
"""
Daily average of Stage
"""
stage_df = water_balance_df[['stage(m)']]
stage_df = stage_df.resample('D', how=np.mean)
# print stage_df.head()
water_balance_daily_df = sum_df.join(stage_df, how='left')
# water_balance_daily_df[water_balance_daily_df['stage(m)'] < stage_cutoff] = 0
water_balance_daily_df['ws_area(sq.m)'] = 0.000
for index, row in water_balance_daily_df.iterrows():
    obs_stage = row['stage(m)']  # observed stage
    if obs_stage >= stage_cutoff:
        x1, x2 = cd.find_range(stage_area_df['stage_m'].tolist(), obs_stage)
        x_diff = x2 - x1
        y1 = stage_area_df['total_area_sq_m'][x1]
        y2 = stage_area_df['total_area_sq_m'][x2]
        y_diff = y2 - y1
        slope = y_diff / x_diff
        y_intercept = y2 - (slope * x2)
        water_balance_daily_df.loc[index.strftime(date_format), 'ws_area(sq.m)'] = (slope * obs_stage) + y_intercept
"""
Change in storage
"""
# separate out 23:30 readings
hour = water_balance_df.index.hour
minute = water_balance_df.index.minute
ch_storage_df = water_balance_df[['volume (cu.m)']][((hour == 23) & (minute == 30))]
ch_storage_df = ch_storage_df.resample('D', how=np.mean)
water_balance_daily_df['change_storage(cu.m)'] = 0.000
for index in ch_storage_df.index:
    if index > min(ch_storage_df.index):
        previous_date = index - timedelta(days=1)
        d1_storage = ch_storage_df['volume (cu.m)'][previous_date.strftime(daily_format)]
        d2_storage = ch_storage_df['volume (cu.m)'][index.strftime(daily_format)]
        water_balance_daily_df.loc[index.strftime(date_format), 'change_storage(cu.m)'] = d2_storage - d1_storage


# new_df = water_balance_daily_df.join(ch_storage_df, how='right')
# new_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/proof.csv')
print water_balance_daily_df.head()

# for index in water_balance_daily_df.index:
#     if index > min(water_balance_daily_df.index):
#         previous_time = index - timedelta(hours=3)
#         d1_storage = water_balance_df['volume (cu.m)'][previous_time.strftime(date_format)]
#         d2_storage = water_balance_df['volume (cu.m)'][index.strftime(date_format)]
#         water_balance_daily_df['change_storage(cu.m)'][index.strftime(date_format)] = d2_storage - d1_storage
print "change_Storage"
"""
Separate out no inflow/ non rainy days
two continuous days of no rain
"""
water_balance_daily_df['status'] = "Y"
for index in water_balance_daily_df.index:
    initial_time_stamp = min(water_balance_daily_df.index) + timedelta(days=1)
    if index > initial_time_stamp and (abs(water_balance_daily_df['change_storage(cu.m)'][index.strftime(daily_format)]) > water_balance_daily_df['Evaporation (cu.m)'][index.strftime(daily_format)])and (water_balance_daily_df["change_storage(cu.m)"][index.strftime(daily_format)] < 0) and (water_balance_daily_df['overflow(cu.m)'][index.strftime(daily_format)] == 0):
    # start_date = index - timedelta(days=1)
    # two_days_rain_df = water_balance_daily_df['Rain Collection (mm)'][
    #                    start_date.strftime(daily_format):index.strftime(daily_format)]
    # sum_df = two_days_rain_df.sum(axis=0)
        water_balance_daily_df['status'][index.strftime(daily_format)] = "N"

dry_water_balance_df = water_balance_daily_df[water_balance_daily_df['status'] == "N"]
rain_water_balance_df = water_balance_daily_df[water_balance_daily_df['status'] == "Y"]
print "dry day sep"
"""
Calculate infiltration
"""
dry_water_balance_df['infiltration(cu.m)'] = 0.000
delta_s = water_balance_daily_df['change_storage(cu.m)']
evap = water_balance_daily_df['Evaporation (cu.m)']
outflow = water_balance_daily_df['overflow(cu.m)']
for index, row in dry_water_balance_df.iterrows():
    if index > min(water_balance_daily_df.index):
        t_1 = index - timedelta(days=1)
        if t_1 < max(water_balance_daily_df.index):
            dry_water_balance_df.loc[index.strftime(date_format), 'infiltration(cu.m)'] = -1.0 * (
                delta_s[index.strftime(daily_format)] + evap[index.strftime(daily_format)])

dry_water_balance_df.loc[:, 'infiltration(cu.m)'] = cd.myround(dry_water_balance_df['infiltration(cu.m)'], decimals=2)
dry_water_balance_df = dry_water_balance_df.loc[dry_water_balance_df['stage(m)'] > 0.1]
dry_water_balance_df = dry_water_balance_df.loc[dry_water_balance_df['infiltration(cu.m)'] > 1]
# dry_water_balance_df = dry_water_balance_df.loc[dry_water_balance_df['infiltration(cu.m)'] < 60]
dry_water_balance_df['infiltration_rate(m)'] = dry_water_balance_df['infiltration(cu.m)']/dry_water_balance_df['ws_area(sq.m)']
print dry_water_balance_df.head()
print dry_water_balance_df['infiltration_rate(m)'].mean()
average_infiltration_rate = cd.myround(dry_water_balance_df['infiltration_rate(m)'].mean(), decimals=3)
print "infilt"
dry_water_balance_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/dry_wb.csv')


"""
Fitting exponential function

stage_cal = dry_water_balance_df['stage(m)']
# stage_cal = dry_water_balance_df['average_stage_m']
inf_cal = dry_water_balance_df['infiltration(cu.m)']
# fig = plt.figure()
# plt.plot(stage_cal, inf_cal, 'bo')
# plt.hlines(y=0, xmin=min(stage_cal), xmax=max(stage_cal))
# plt.show()
# print dry_water_balance_df.shape
log_x = np.log(stage_cal)
log_y = np.log(inf_cal)

OK = log_y == log_y
masked_log_y = log_y[OK]
masked_log_x = log_x[OK]
# pars = np.polyfit(masked_log_x, masked_log_y, 1)
slope, intercept = ccy_classic_lstsqr.ccy_classic_lstsqr(masked_log_x, masked_log_y)
print "fit done"
print slope, intercept
"""

"""
Rainy day infiltration
"""
rain_water_balance_df['infiltration(cu.m)'] = 0.0
for i in rain_water_balance_df.index:
    if rain_water_balance_df.loc[i.strftime(daily_format), 'stage(m)'] >= stage_cutoff:
        surface_area = rain_water_balance_df.loc[i.strftime(daily_format), 'ws_area(sq.m)']
        rain_water_balance_df.loc[i.strftime(daily_format), 'infiltration(cu.m)'] = average_infiltration_rate*surface_area
print "rainy day"
"""
Inflow calculation
"""
merged_water_balance = pd.concat([dry_water_balance_df, rain_water_balance_df])
merged_water_balance['Inflow (cu.m)'] = 0.000
delta_s_rain = water_balance_daily_df['change_storage(cu.m)']
inf_rain = merged_water_balance['infiltration(cu.m)']
evap_rain = water_balance_daily_df['Evaporation (cu.m)']
outflow_rain = water_balance_daily_df['overflow(cu.m)']
for i, row in merged_water_balance.iterrows():
    if i > min(merged_water_balance.index):
        string1 = intern(row['status'])
        string2 = intern('N')
        if string1 != string2:
            # i_1 = i - timedelta(days=1)
            merged_water_balance.loc[i.strftime(daily_format), 'Inflow (cu.m)'] = (delta_s_rain[i.strftime(daily_format)] +
                                                                              inf_rain[i.strftime(daily_format)] +
                                                                              evap_rain[i.strftime(daily_format)] +
                                                                              outflow_rain[i.strftime(daily_format)])

merged_water_balance.sort_index(inplace=True)
wb = (merged_water_balance['Evaporation (cu.m)'].sum() +
      merged_water_balance['infiltration(cu.m)'].sum() +
      merged_water_balance['overflow(cu.m)'].sum()) - merged_water_balance['Inflow (cu.m)'].sum()

print "E =", merged_water_balance['Evaporation (cu.m)'].sum()
print "Infil=", merged_water_balance['infiltration(cu.m)'].sum()
print "Overflow=", merged_water_balance['overflow(cu.m)'].sum()
print "Inflow =", merged_water_balance['Inflow (cu.m)'].sum()
print "Storage=", wb

merged_water_balance.index.name = 'Date'
merged_water_balance['cum_rain'] = merged_water_balance['rain (mm)'].cumsum()
print merged_water_balance.dtypes
merged_water_balance['Inflow (cu.m)'] = merged_water_balance['Inflow (cu.m)'].astype(float)
merged_water_balance.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/et_infilt_634_w_of.csv')
new_df = merged_water_balance.join(ch_storage_df, how='right')
new_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/proof.csv')
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, sharex=True)
bar2 = ax2.bar(merged_water_balance.index, merged_water_balance['Inflow (cu.m)'], color='b')
bar3 = ax3.bar(merged_water_balance.index, merged_water_balance['infiltration(cu.m)'], color='r')
bar4 = ax4.bar(merged_water_balance.index, merged_water_balance['Evaporation (cu.m)'], color='g')
bar1 = ax1.plot(water_balance_df.index, water_balance_df['stage(m)'], 'ro-', label='Stage (m)' )
bar5 = ax5.bar(merged_water_balance.index, merged_water_balance['rain (mm)'], color='y')
# plt.plot(merged_water_balance['Inflow (cu.m)'],merged_water_balance['cum_rain'], 'bo')
# plt.plot(merged_water_balance.index, merged_water_balance['infiltration(cu.m)'], '-ro')
# ax4.hlines(y=0, xmin=min(merged_water_balance.index), xmax=max(merged_water_balance.index))
ax1.legend(prop={'size':16}).draggable()
lns = [bar2, bar3, bar4, bar5]
labs =['Inflow (cu.m)', "Infiltration (cu.m)", "Evaporation (cu.m)", 'Rain (mm)']
# Shrink current axis's height by 10% on the bottom
box = ax5.get_position()
ax5.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
ax5.legend(lns, labs, loc='upper center', fancybox=True, ncol=4, bbox_to_anchor=(0.5, -0.05),prop={'size':16} ).draggable()
ax1.set_title("Check dam 634")
plt.show()
