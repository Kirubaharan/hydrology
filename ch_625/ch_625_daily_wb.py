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

"""
Variables
"""
full_stage = 0.61  # check dam height, above this it is assumed check dam will overflow
date_format = '%Y-%m-%d %H:%M:%S'
daily_format = '%Y-%m-%d'
resolution_ody = 0.0008
stage_cutoff = 0.1

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
print rain_df.head()
raise SystemExit(0)
"""
Check dam calibration
"""
y_cal = np.array([100, 400, 1000, 1600, 2250, 2750])
x_cal = np.array([1987, 2454, 3344, 4192, 5104, 5804])
a_stage = cd.polyfit(x_cal, y_cal, 1)
coeff_cal = a_stage['polynomial']
slope = coeff_cal[0]
intercept = coeff_cal[1]
"""
Read Check dam data
"""
block_1 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2526/2526_007_001.CSV'
water_level_1 = cd.read_correct_ch_dam_data(block_1, slope, intercept)
block_2 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2526/2526_007_002_25_8_14.CSV'
water_level_2 = cd.read_correct_ch_dam_data(block_2, slope, intercept)
block_3 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2526/2526_007_003.CSV'
water_level_3 = cd.read_correct_ch_dam_data(block_3, slope, intercept)
block_4 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2526/2526_007_004.CSV'
water_level_4 = cd.read_correct_ch_dam_data(block_4, slope, intercept)
block_5 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2526/2526_005_001.CSV'
water_level_5 = cd.read_correct_ch_dam_data(block_5, slope, intercept)
block_6 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2526/2526_005_002_03_12_2014.CSV'
water_level_6 = cd.read_correct_ch_dam_data(block_6, slope, intercept)

for i in range(1, 7, 1):
    eval("water_level_{0}.drop(water_level_{0}.tail(1).index, inplace=True, axis=0)".format(i))
    eval("water_level_{0}.drop(water_level_{0}.head(1).index, inplace=True, axis=0)".format(i))

# for i in range(1, 7, 1):
#     print "water_level_{0}".format(i)
#     print eval("water_level_{0}.head()".format(i))

fig = plt.figure()
for i in range(1, 7, 1):
    x = eval("water_level_{0}.index".format(i))
    y = eval("water_level_{0}['stage(m)']".format(i))
    plt.plot(x, y)

plt.show()
# 30 min data
water_level_30min = pd.concat([water_level_1, water_level_2, water_level_3, water_level_4], axis=0)
water_level_30min = water_level_30min.sort()
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
# 10 min data
water_level_10min = pd.concat([water_level_5, water_level_6], axis=0)
water_level_10 = water_level_10min.sort()
rounded = np.array(water_level_10min.index, dtype='datetime64[m]')
water_level_10min = water_level_10min.set_index(rounded)
start_time_10 = min(water_level_10min.index)
end_time_10 = max(water_level_10min.index)
new_index_10 = pd.date_range(start=start_time_10, end=end_time_10, freq='10min')
water_level_10min = water_level_10min.reindex(new_index_10, method=None)
water_level_10min = water_level_10min.interpolate(method='time')
water_level_10min.index.name = 'Date'
water_level = pd.concat([water_level_30min, water_level_10min], axis=0)
water_level = water_level.resample('30min', how=np.mean, label='right', closed='right')
water_level.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_625/stage_625.csv')
water_level[water_level['stage(m)'] < stage_cutoff] = 0
"""
Join weather and rain data
"""
weather_df = weather_df.join(rain_df, how='right')
weather_df = weather_df[min(water_level.index).strftime(daily_format): max(water_level.index).strftime(daily_format)]
weather_df = weather_df.join(water_level, how='right')
"""
Remove Duplicates
"""
weather_df['index'] = weather_df.index
weather_df.drop_duplicates(subset='index', take_last=True, inplace=True)
del weather_df['index']
weather_df = weather_df.sort()
"""
Open water evaporation
"""
z = 830
p = (1 - (2.25577 * (10 ** -5) * z))
air_p_pa = 101325 * (p ** 5.25588)
# give air pressure value
weather_df['AirPr(Pa)'] = air_p_pa
"""
Half hourly Extraterrestrial Radiation Calculation(J/m2/30min)
"""
sc_default = 1367.0  # Solar constant in W/m^2 is 1367.0.
ch_625_lat = 13.364112
ch_625_long = 77.556057

weather_df['Rext (MJ/m2/30min)'] = 0.000
for i in weather_df.index:
    weather_df['Rext (MJ/m2/30min)'][i.strftime(date_format)] = (
        cd.extraterrestrial_irrad(local_datetime=i, latitude_deg=ch_625_lat, longitude_deg=ch_625_long))

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
raise SystemExit(0)
"""
Convert observed stage to volume by linear interpolation
"""
# set stage as index
stage_vol_df = pd.read_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_625/stage_vol.csv',
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
        water_balance_df['volume (cu.m)'][index.strftime('%Y-%m-%d %H:%M:%S')] = (slope * obs_stage) + y_intercept

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
