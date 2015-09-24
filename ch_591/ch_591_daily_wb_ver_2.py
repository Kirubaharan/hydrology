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

print pd.__version__

"""
Variables
"""
full_stage = 1.96  # check dam height, above this it is assumed check dam will overflow
date_format = '%Y-%m-%d %H:%M:%S'
daily_format = '%Y-%m-%d'
resolution_ody = 0.0008
stage_cutoff = 0.1

# ------------------------------------------------------------------#
# Weather file
weather_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/ksndmc_weather.csv'
# Rain file
rain_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/ksndmc_rain.csv'
# convert to pandas dataframe
weather_df = pd.read_csv(weather_file, sep=',', header=0)
# print weather_df.head()
weather_df['Date_Time'] = pd.to_datetime(weather_df['Date_Time'], format=date_format)
weather_df.set_index(weather_df['Date_Time'], inplace=True)
weather_df.sort_index(inplace=True)
weather_df = weather_df.drop('Date_Time', 1)

# Rain data frame
rain_df = pd.read_csv(rain_file, sep=',', header=0)
# set index
rain_df['Date_Time'] = pd.to_datetime(rain_df['Date_Time'], format=date_format)
rain_df.set_index(rain_df['Date_Time'], inplace=True)
# sort based on index
rain_df.sort_index(inplace=True)
# drop date time column
rain_df = rain_df.drop('Date_Time', 1)
print rain_df.head()
# rain_df_hourly_sum = rain_df.resample('H', how=np.sum)
# rain_df_hourly_mean = rain_df.resample('H', how=np.mean)
# rain_df_hourly_sum.to_csv('/media/kiruba/New Volume/ACCUWA_Data/hourly_rain_sum_aral.csv')
# rain_df_hourly_mean.to_csv('/media/kiruba/New Volume/ACCUWA_Data/hourly_rain_mean_aral.csv')

raise SystemExit(0)
"""
Check dam calibration
"""
y_cal = np.array([100, 400, 1000, 1600, 2250, 2750, 3000])
x_cal = np.array([2036, 2458, 3025, 4078, 5156, 5874, 6198])
a_stage = cd.polyfit(x_cal, y_cal, 1)
coeff_cal = a_stage['polynomial']
slope = coeff_cal[0]
intercept = coeff_cal[1]
print coeff_cal
# # 2665
y_cal_1 = [100, 400, 1000, 1500, 2000, 3000]
x_cal_1 = [1971, 2336, 3083, 3720, 4335, 5604]
coeff = np.polyfit(x_cal_1, y_cal_1,1)
print coeff
# print coeff[0]*3020 + coeff[1]
# print coeff[0]*2810 + coeff[1]
# raise SystemExit(0)
fig = plt.figure()
plt.plot(x_cal_1, y_cal_1, 'ro-')
plt.show()

"""
Read Check dam data
"""
block_1 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525/2525_008_001.CSV'
water_level_1 = cd.read_correct_ch_dam_data(block_1, slope, intercept)
block_2 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525/2525_008_002.CSV'
water_level_2 = cd.read_correct_ch_dam_data(block_2, slope, intercept)
block_3 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525/2525_008_003.CSV'
water_level_3 = cd.read_correct_ch_dam_data(block_3, slope, intercept)
block_4 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525/2525_008_004.CSV'
water_level_4 = cd.read_correct_ch_dam_data(block_4, slope, intercept)
block_5 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525/2525_008_005.CSV'
water_level_5 = cd.read_correct_ch_dam_data(block_5, slope, intercept)
block_6 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525/2525_008_006.CSV'
water_level_6 = cd.read_correct_ch_dam_data(block_6, slope, intercept)
block_7 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525/2525_004_001.CSV'
water_level_7 = cd.read_correct_ch_dam_data(block_7, slope, intercept)
# water_level_7['stage(m)'] += 0.05
block_8 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525/2525_004_002_12_12_2014.CSV'
water_level_8 = cd.read_correct_ch_dam_data(block_8, slope, intercept)
# water_level_8['stage(m)'] += 0.1
block_9 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525/2525_004_003_16_12_2014.CSV'
water_level_9 = cd.read_correct_ch_dam_data(block_9, slope, intercept)
block_10 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525/2525_004_004_24_12_2014.CSV'
water_level_10 = cd.read_correct_ch_dam_data(block_10, slope, intercept)
block_11 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525/2525_004_005_24_01_2015.CSV'
water_level_11 = cd.read_correct_ch_dam_data(block_11, slope, intercept)
block_12 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525/2525_004_006_10_02_2015.CSV'
water_level_12 = cd.read_correct_ch_dam_data(block_12, slope, intercept)
block_13 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525/2525_008_007_02_03_2015.CSV'
water_level_13 = cd.read_correct_ch_dam_data(block_13, slope, intercept)
block_14 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2665/check/2665_006_001.CSV-05-08-2015.CSV'
water_level_14 = cd.read_correct_ch_dam_data(block_14, coeff[0], coeff[1])
block_15 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2665/check/2665_006_002.CSV-17-08-15.CSV'
water_level_15 = cd.read_correct_ch_dam_data(block_15, coeff[0], coeff[1])
block_16 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2665/check/2665_006_003.CSV-8-9-15.CSV'
water_level_16 = cd.read_correct_ch_dam_data(block_16, coeff[0], coeff[1])
block_17 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2665/check/2665_006_004_18_09_2015.CSV'
water_level_17 = cd.read_correct_ch_dam_data(block_17, coeff[0], coeff[1])
block_18 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2665/check/2665_006_005.CSV'
water_level_18 = cd.read_correct_ch_dam_data(block_18, coeff[0], coeff[1])


for i in range(1, 19, 1):
    eval("water_level_{0}.drop(water_level_{0}.tail(1).index, inplace=True, axis=0)".format(i))
    eval("water_level_{0}.drop(water_level_{0}.head(1).index, inplace=True, axis=0)".format(i))

# for i in range(1, 11, 1):
#     print "water_level_{0}".format(i)
#     print eval("water_level_{0}.head()".format(i))
fig = plt.figure()
for i in range(1, 19, 1):
    x = eval("water_level_{0}.index".format(i))
    y = eval("water_level_{0}['stage(m)']".format(i))
    plt.plot(x, y)

plt.show()
# print water_level_13.head()
raise SystemExit(0)

water_level_30min = pd.concat([water_level_1, water_level_2, water_level_3, water_level_4, water_level_5], axis=0)
water_level_30 = water_level_30min.sort()
rounded = np.array(water_level_30min.index, dtype='datetime64[m]')
water_level_30min = water_level_30min.set_index(rounded)
start_time_30 = min(water_level_30min.index)
end_time_30 = max(water_level_30min.index)
# new_index_30min = pd.date_range(start=start_time_30.strftime('%Y-%m-%d %H:%M'), end=end_time_30.strftime('%Y-%m-%d %H:%M'), freq='30min')
# if type(new_index_30min) is list:
#     print "list"
# else:
#     print "no"
# raise SystemExit(0)
new_index_30 = pd.date_range(start=start_time_30, end=end_time_30, freq='30min')
water_level_30min = water_level_30min.reindex(new_index_30, method=None)
water_level_30min = water_level_30min.interpolate(method='time')
# water_level_30min = water_level_30min.set_index(new_index_30min)
water_level_30min.index.name = 'Date'
# print water_level_30min.head()
# raise SystemExit(0)
water_level_10min = pd.concat([water_level_6, water_level_7, water_level_8, water_level_9, water_level_10, water_level_11, water_level_12], axis=0)
water_level_10min = water_level_10min.sort()
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
water_level = water_level[:'2015-02-09']
print water_level.tail()
# raise SystemExit(0)
# water_level['stage(m)'] = cd.myround(water_level['stage(m)'], decimals=2)
water_level.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/stage_591.csv')
hour = water_level.index.hour
minute = water_level.index.minute
eleven_30_df = water_level[((hour == 23) & (minute == 30))]
print eleven_30_df.head()
fig = plt.figure()
plt.plot(eleven_30_df.index, eleven_30_df['raw value'], 'ro')
plt.show()
# raise SystemExit(0)
# raise SystemExit(0)
"""
Join weather and rain data
"""
weather_df = weather_df.join(rain_df, how='right')
weather_df = weather_df[min(water_level.index).strftime(daily_format): max(water_level.index).strftime(daily_format)]
weather_df = weather_df.join(water_level, how='inner')
# fig, ax1 = plt.subplots()
# ax1.bar(weather_df.index, weather_df['Rain Collection (mm)'], 0.35, color='b')
# plt.gca().invert_yaxis()
# ax2 = ax1.twinx()
# ax2.plot(weather_df.index, weather_df['stage(m)'], '-b')
# plt.hlines(y=stage_cutoff, xmin=min(weather_df.index), xmax=max(weather_df.index))
# plt.hlines(y=full_stage, xmin=min(weather_df.index), xmax=max(weather_df.index), colors='g')
# plt.show()
# raise SystemExit(0)
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
z = 799
p = (1 - (2.25577 * (10 ** -5) * z))
air_p_pa = 101325 * (p ** 5.25588)
# give air pressure value
weather_df['AirPr(Pa)'] = air_p_pa
# print weather_df.head()
# raise SystemExit(0)
"""
Half hourly Extraterrestrial Radiation Calculation(J/m2/30min)
"""
sc_default = 1367.0  # Solar constant in W/m^2 is 1367.0.
ch_591_lat = 13.260196
ch_591_long = 77.512085
weather_df['Rext (MJ/m2/30min)'] = 0.000
for i in weather_df.index:
    weather_df['Rext (MJ/m2/30min)'][i.strftime(date_format)] = (cd.extraterrestrial_irrad(local_datetime=i,
                                                                                                   latitude_deg=ch_591_lat,
                                                                                                   longitude_deg=ch_591_long))
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

print  weather_df.head(10)
print weather_df.tail(10)
# raise SystemExit(0)
"""
Half hourly Evaporation calculation
"""
# airtemp = weather_df['Air Temperature (C)']
airtemp = weather_df['TEMPERATURE']
hum = weather_df['Humidity (%)']
airpress = weather_df['AirPr(Pa)']
rs = weather_df['Solar Radiation (MJ/m2/30min)']
rext = weather_df['Rext (MJ/m2/30min)']
# wind_speed = weather_df['Wind Speed (mps)']
wind_speed = weather_df['WIND_SPEED']
weather_df['Evaporation (mm/30min)'] = cd.half_hour_evaporation(airtemp=airtemp, rh=hum, airpress=airpress,
                                                                rs=rs, rext=rext, u=wind_speed, z=z)

weather_df_monthly = weather_df.resample('M', how=np.sum)
weather_df_monthly.to_csv('/home/kiruba/Documents/et_monthly.csv')
# raise SystemExit(0)

"""
Select data where stage is available
"""
weather_stage_avl_df = weather_df[min(water_level.index):max(water_level.index)]
"""
Convert observed stage to volume by linear interpolation
"""
# set stage as index
stage_vol_df = pd.read_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/stage_vol_new.csv',
                           sep=',', header=0, names=['sno', 'stage_m', 'total_vol_cu_m'])
# print stage_vol_df

stage_vol_df.drop('sno', inplace=True, axis=1)
stage_vol_df.set_index(stage_vol_df['stage_m'], inplace=True)
water_balance_df = weather_stage_avl_df[['rain (mm)', 'Evaporation (mm/30min)', 'stage(m)']]
water_balance_df['volume (cu.m)'] = 0.000
print water_balance_df.head()

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
        water_balance_df['volume (cu.m)'][index.strftime(date_format)] = (slope * obs_stage) + y_intercept

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
print("full volume = %s" % full_volume)
# overflow_check_df = water_balance_df['2014-08-21']
# fig = plt.figure()
# plt.plot_date(overflow_check_df.index, overflow_check_df['stage(m)'], 'bo-')
# plt.hlines(y=full_stage, xmin=min(overflow_check_df.index), xmax=max(overflow_check_df.index))
# plt.show()
# raise SystemExit(0)

"""
Overflow
"""
water_balance_df['overflow(cu.m)'] = 0.000
# length_check_dam = 16.85 # b
# cd_wall_length_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/ch_591_wall_elevation.csv'
# cd_wall_length_df = pd.read_csv(cd_wall_length_file, sep=',', header=0)
# cd_wall_length_df.set_index(cd_wall_length_df['elevation'], inplace=True)
# coeff_discharge = 0.6
#
# for index, row in water_balance_df.iterrows():
#     obs_stage = cd.myround(a=row['stage(m)'], decimals=2)
#     if obs_stage > full_stage:
#         effective_head = obs_stage - full_stage
#         previous_time = index - timedelta(minutes=30)
#         previous_stage = cd.myround(a=water_balance_df['stage(m)'][previous_time.strftime(date_format)], decimals=2)
#         if previous_stage > full_stage:
#             print full_stage, previous_stage
#             overflow_volume = 0.0
#             effective_head_1 = previous_stage - full_stage
#             diff_eff_head = list(cd.spread(effective_head_1, effective_head, 1800, mode=3))
#             # print diff_eff_head
#             for eff_head in diff_eff_head:
#                 print 'eff head  =  %0.04f'  % eff_head
#                 true_head = cd.myround(a=(eff_head + full_stage), decimals=2)
#                 if true_head < max(cd_wall_length_df.index):
#                     effective_length = cd_wall_length_df['length'][true_head]
#                 else:
#                     effective_head = length_check_dam
#                 d_vol = (2.0/3.0)*coeff_discharge*effective_length*(math.sqrt(2.*9.81))*(eff_head**1.5)
#                 print d_vol
#                 print overflow_volume
#                 overflow_volume += d_vol
#             water_balance_df['overflow(cu.m)'][index.strftime(date_format)] = overflow_volume
#         else:
#             if obs_stage < max(cd_wall_length_df.index):
#                 effective_length = cd_wall_length_df['length'][obs_stage]
#             else:
#                 effective_length = length_check_dam
#             x1 = 0
#             x2 = 1800
#             y1 = cd.myround(a=water_balance_df['stage(m)'][previous_time.strftime(date_format)], decimals=2)
#             y2 = cd.myround(a=obs_stage, decimals=2)
#             slope = (y1 - y2) / (x1 - x2)
#             intercept = y2 - (slope * x2)
#             time_of_overflow = 1800 - ((full_stage - intercept) / slope)
#             water_balance_df['overflow(cu.m)'][index.strftime(date_format)] = time_of_overflow*(2.0/3.0)*coeff_discharge*effective_length*(math.sqrt(2.*9.81))*(effective_head**1.5)

for index, row in water_balance_df.iterrows():
    obs_volume = row['volume (cu.m)']
    if obs_volume > full_volume:
        overflow_volume = obs_volume - full_volume
        water_balance_df[ 'overflow(cu.m)'][index.strftime(date_format)] = overflow_volume

print water_balance_df['overflow(cu.m)'].sum()

water_balance_df = water_balance_df["2014-05-15":]
water_balance_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/overflow_check.csv')


"""
Overflow
# length_check_dam = 16.85 # b
# width_check_dam = 0.55   # l
# no_of_contractions = 0
# coeff_discharge = 0.6
water_balance_df['overflow(cu.m)'] = 0.000
# water_balance_df['overflow_rate(cu.mps)'] = 0.000
# cd_wall_length_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/ch_591_wall_elevation.csv'
# cd_wall_length_df = pd.read_csv(cd_wall_length_file, sep=',', header=0)
# cd_wall_length_df.set_index(cd_wall_length_df['elevation'], inplace=True)
# print cd_wall_length_df
# raise SystemExit(0)
for index, row in water_balance_df.iterrows():
    obs_stage = cd.myround(a=row['stage(m)'], decimals=2)
    if obs_stage > full_stage:
        effective_head = obs_stage - full_stage
        print width_check_dam/effective_head
        previous_time = index - timedelta(seconds=1800)
        previous_stage = cd.myround(a=water_balance_df['stage(m)'][previous_time.strftime(date_format)], decimals=2)
        if obs_stage < max(cd_wall_length_df.index):
            effective_length = cd_wall_length_df['length'][obs_stage]
        else:
            effective_length = length_check_dam
        print obs_stage, effective_length
        water_balance_df['overflow_rate(cu.mps)'][index.strftime(date_format)] = (2.0/3.0)*coeff_discharge*effective_length*(math.sqrt(2.*9.81))*(effective_head**1.5)
        if  previous_stage > full_stage:
            water_balance_df['overflow(cu.m)'][index.strftime(date_format)] = full_vol - (1800*water_balance_df['overflow_rate(cu.mps)'][index.strftime(date_format)])


        else:
            x1 = 0
            x2 = 900
            y1 = cd.myround(a=water_balance_df['stage(m)'][previous_time.strftime(date_format)], decimals=2)
            y2 = cd.myround(a=obs_stage, decimals=2)
            slope = (y1 - y2) / (x1 - x2)
            intercept = y2 - (slope * x2)
            time_of_overflow = 1800 - ((full_stage - intercept) / slope)
            water_balance_df['overflow(cu.m)'][
                index.strftime(date_format)] = full_vol - (time_of_overflow*water_balance_df['overflow_rate(cu.mps)'][index.strftime(date_format)])
water_balance_df = water_balance_df["2014-10-08":"2014-10-09"]
print water_balance_df['overflow(cu.m)'].sum()
print "overflow"
water_balance_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/overflow_check.csv')
raise SystemExit(0)
# for index in water_balance_df:
#     if index > min(water_balance_df.index):
"""

"""
Stage vs area linear relationship
"""
stage_area_df = pd.read_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/cont_area.csv',
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
        water_balance_df['ws_area(sq.m)'][index.strftime(date_format)] = (slope * obs_stage) + y_intercept
# print water_balance_df.head()
# raise SystemExit(0)
"""
Surface area to volume ratio
"""
# average_stage = full_stage/2.0
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
print("full surface area = %s" % average_area)
surface_area_to_vol_ratio = average_area/average_volume
print 'surface area to vol ratio is %0.2f' % surface_area_to_vol_ratio
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
print stage_df.head()
water_balance_daily_df = sum_df.join(stage_df, how='left')

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
        water_balance_daily_df['ws_area(sq.m)'][index.strftime(date_format)] = (slope * obs_stage) + y_intercept
"""
Change in storage
"""
# water_balance_daily_df = water_balance_df
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
        water_balance_daily_df['change_storage(cu.m)'][index.strftime(daily_format)] = d2_storage - d1_storage

# fig = plt.figure()
# plt.plot(water_balance_daily_df.index, water_balance_daily_df['change_storage(cu.m)'], '-ro')
# plt.show()
# raise SystemExit(0)

# new_df = water_balance_daily_df.join(ch_storage_df, how='right')
# new_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/proof.csv')

# print water_balance_df.head()

# for index in water_balance_daily_df.index:
#     if index > min(water_balance_daily_df.index):
#         previous_time = index - timedelta(minutes=30)
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
    if index > initial_time_stamp and (water_balance_daily_df["change_storage(cu.m)"][index.strftime(daily_format)] < 0) and (water_balance_daily_df['overflow(cu.m)'][index.strftime(daily_format)] == 0) and (abs(water_balance_daily_df['change_storage(cu.m)'][index.strftime(daily_format)]) > water_balance_daily_df['Evaporation (cu.m)'][index.strftime(daily_format)]):
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
            dry_water_balance_df['infiltration(cu.m)'][index.strftime(date_format)] = -1.0 * (
                delta_s[index.strftime(daily_format)] + evap[index.strftime(daily_format)])

# dry_water_balance_df[:, 'infiltration(cu.m)'] = cd.myround(dry_water_balance_df['infiltration(cu.m)'], decimals=3)
dry_water_balance_df = dry_water_balance_df[dry_water_balance_df['stage(m)'] > 0.1]
dry_water_balance_df = dry_water_balance_df[dry_water_balance_df['infiltration(cu.m)'] > 1.0]
# dry_water_balance_df = dry_water_balance_df[dry_water_balance_df['infiltration(cu.m)'] < 60]
dry_water_balance_df['infiltration_rate(m)'] = dry_water_balance_df['infiltration(cu.m)']/dry_water_balance_df['ws_area(sq.m)']
dry_water_balance_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/dry_wb.csv')
print dry_water_balance_df.head()
print dry_water_balance_df['infiltration_rate(m)'].mean()
average_infiltration_rate = cd.myround(dry_water_balance_df['infiltration_rate(m)'].mean(), decimals=3)
print "infilt"
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, facecolor='white', sharex=True)
ax1.plot(dry_water_balance_df.index, dry_water_balance_df['infiltration_rate(m)'], 'ro')
ax2.plot(dry_water_balance_df.index, dry_water_balance_df['infiltration(cu.m)'], 'bo')
ax3.plot(dry_water_balance_df.index, dry_water_balance_df['Evaporation (cu.m)'], 'go')
ax4.plot(eleven_30_df.index, eleven_30_df['raw value'], 'ko')
plt.show()
# raise SystemExit(0)

"""
Fitting exponential function

dry_water_balance_df['average_stage(m)'] = 0.00

# for index in dry_water_balance_df.index:
#     if index > min(dry_water_balance_df.index):
#         previous_time = index - timedelta(minutes=30)
#         previous_stage = dry_water_balance_df['stage(m)'][previous_time.strftime(date_format)]
#         dry_water_balance_df['average_stage(m)'][index.strftime(date_format)] = 0.5*(dry_water_balance_df['stage(m)'][index.strftime(date_format)] + previous_stage)

stage_cal = dry_water_balance_df['stage(m)']
# stage_cal = dry_water_balance_df['average_stage_m']
inf_cal = dry_water_balance_df['infiltration(cu.m)']
# fig = plt.figure()
# plt.plot(stage_cal, inf_cal, 'bo')
# plt.hlines(y=0, xmin=min(stage_cal), xmax=max(stage_cal))
# plt.xlabel('Stage (m)')
# plt.ylabel('Infiltration (cu.m)')
# plt.savefig('/media/kiruba/New Volume/AGU/poster/agu_checkdam/image/stage_inf.png',bbox_inches='tight')
# plt.show()
# raise SystemExit(0)
# print dry_water_balance_df.shape
log_x = np.log(stage_cal)
log_y = np.log(inf_cal)

OK = log_y == log_y
masked_log_y = log_y[OK]
masked_log_x = log_x[OK]
fig = plt.figure()
plt.plot(masked_log_x, masked_log_y, 'ro')
plt.show()
# raise SystemExit(0)
# pars = np.polyfit(masked_log_x, masked_log_y, 1)
slope, intercept = ccy_classic_lstsqr.ccy_classic_lstsqr(masked_log_x, masked_log_y)
print "fit done"
print slope, intercept
print math.exp(slope), math.exp(intercept)
raise SystemExit(0)
"""
"""
Rainy day infiltration
"""
rain_water_balance_df['infiltration(cu.m)'] = 0.0
for i in rain_water_balance_df.index:
    if rain_water_balance_df['stage(m)'][i.strftime(daily_format)] >= stage_cutoff:
        surface_area = rain_water_balance_df['ws_area(sq.m)'][i.strftime(daily_format)]
        rain_water_balance_df['infiltration(cu.m)'][i.strftime(daily_format)] = average_infiltration_rate*surface_area
print "rainy day"
"""
Inflow calculation
"""
merged_water_balance = pd.concat([dry_water_balance_df, rain_water_balance_df])
merged_water_balance = merged_water_balance.sort()
print merged_water_balance.tail()
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
            merged_water_balance['Inflow (cu.m)'][i.strftime(daily_format)] = (delta_s_rain[i.strftime(daily_format)] +
                                                                              inf_rain[i.strftime(daily_format)] +
                                                                              evap_rain[i.strftime(daily_format)] +
                                                                              outflow_rain[i.strftime(daily_format)])

merged_water_balance.sort_index(inplace=True)
merged_water_balance.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/merged_check.csv')

wb = (merged_water_balance['Evaporation (cu.m)'].sum() +
      merged_water_balance['infiltration(cu.m)'].sum() +
      merged_water_balance['overflow(cu.m)'].sum()) - merged_water_balance['Inflow (cu.m)'].sum()

print "E =", merged_water_balance['Evaporation (cu.m)'].sum()
print "Infil=", merged_water_balance['infiltration(cu.m)'].sum()
print "Overflow=", merged_water_balance['overflow(cu.m)'].sum()
print "Inflow =", merged_water_balance['Inflow (cu.m)'].sum()
print "Storage=", wb

merged_water_balance.index.name = 'Date'
# merged_water_balance['cum_rain'] = merged_water_balance['rain (mm)'].cumsum()
print merged_water_balance.dtypes
# raise SystemExit(0)
merged_water_balance['Inflow (cu.m)'] = merged_water_balance['Inflow (cu.m)'].astype(float)
merged_water_balance.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/et_infilt_591_w_of.csv')
new_df = merged_water_balance.join(ch_storage_df, how='right')
new_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/proof.csv')
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, sharex=True, facecolor='white')
bar2 = ax2.bar(merged_water_balance.index, merged_water_balance['Inflow (cu.m)'], color='b')
bar3 = ax3.bar(merged_water_balance.index, merged_water_balance['infiltration(cu.m)'], color='r')
bar4 = ax4.bar(merged_water_balance.index, merged_water_balance['Evaporation (cu.m)'], color='g')
bar1 = ax1.plot(water_balance_df.index, water_balance_df['stage(m)'], 'r-', label='Stage (m)' )
bar5 = ax5.bar(merged_water_balance.index, merged_water_balance['rain (mm)'], color='y')
# plt.plot(merged_water_balance['Inflow (cu.m)'],merged_water_balance['cum_rain'], 'bo')
# plt.plot(merged_water_balance.index, merged_water_balance['infiltration(cu.m)'], '-ro')
# ax4.hlines(y=0, xmin=min(merged_water_balance.index), xmax=max(merged_water_balance.index))
ax1.legend(prop={'size':16}).draggable()
lns = [bar2, bar3, bar4, bar5]
labs =['Inflow (cu.m)', "Percolation (cu.m)", "Evaporation (cu.m)", 'Rain (mm)']
# Shrink current axis's height by 10% on the bottom
box = ax5.get_position()
ax5.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
ax5.legend(lns, labs, loc='upper center', fancybox=True, ncol=4, bbox_to_anchor=(0.5, -0.05),prop={'size':16} ).draggable()
ax1.set_title("Check dam 591")
plt.show()
