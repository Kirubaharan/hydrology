__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import checkdam.checkdam as cd
from datetime import timedelta
import ccy_classic_lstsqr
from bokeh.plotting import figure, show, output_file
from bokeh.models import LinearAxis, Range1d
from bokeh.charts import Bar, output_file, show


"""
Variables
"""
# lowest boundary 902.873, elevation at measuring site 902.166809 - stage at july 14 2015
full_stage = 3.8  # meter, so for max stage is 3.407
date_format = '%Y-%m-%d %H:%M:%S'
daily_format = '%Y-%m-%d'
stage_cutoff = 0.1

# tank water level

# # rain_file
# rain_file = '/media/kiruba/New Volume/milli_watershed/cumulative impacts/had_rainfall.csv'
# rain_df = pd.read_csv(rain_file, sep=',', header=0)
# # set index
# rain_df['Date_Time'] = pd.to_datetime(rain_df['date_time'], format=date_format)
# rain_df.set_index(rain_df['Date_Time'], inplace=True)
# # sort based on index
# rain_df.sort_index(inplace=True)
# # drop date time column
# rain_df = rain_df.drop('Date_Time', 1)
# rain_df['index'] = rain_df.index
# rain_df.drop_duplicates(subset='index', keep='last', inplace=True)
# del rain_df['index']
# rain_df.sort_index(inplace=True)
# rain_df = rain_df[min(model_water_level.index):]

water_level_file = '/media/kiruba/New Volume/milli_watershed/tmg_lake_water_balance/water_level_had_tank.csv'
water_level_df = pd.read_csv(water_level_file, sep=',')
water_level_df['date_time'] = pd.to_datetime(water_level_df['date_time'], format=date_format)
water_level_df.set_index(water_level_df['date_time'], inplace=True)
water_level_df.drop('date_time', 1, inplace=True)
# print(np.max(water_level_df['stage(m)']))
weather_file = '/media/kiruba/New Volume/milli_watershed/tmg_lake_water_balance/tmg_open_water_evap.csv'
weather_df = pd.read_csv(weather_file, sep=',')
weather_df.drop('Unnamed: 0', 1, inplace=True)
weather_df['date_time'] = pd.to_datetime(weather_df['date_time'], format=daily_format)
weather_df.set_index(weather_df['date_time'], inplace=True)
weather_df.drop('date_time', 1, inplace=True)
"""
Select data where stage is available
"""
weather_df = weather_df[min(water_level_df.index):max(water_level_df.index)]
#  select only 11 30 pm values of stage
hour = water_level_df.index.hour
minute = water_level_df.index.minute
eleven_30_df = water_level_df[((hour == 23) & (minute == 30))]
eleven_30_df = eleven_30_df.resample('D', how=np.mean)
# join 23 30 water level to the weather df
weather_df = weather_df.join(eleven_30_df, how='inner')
water_level__daily_df = water_level_df.resample('D', how=np.mean)
weather_df.loc[:, 'mean_stage_m'] = water_level__daily_df.loc[:, 'stage(m)']
# print weather_df.head()

"""
Convert observed stage to volume by linear interpolation
"""
# set stage as index
stage_vol_df = pd.read_csv('/media/kiruba/New Volume/milli_watershed/tmg_lake_bathymetry/stage_volume_area/stage_volume_tmg.csv',
                           sep=',', header=0, names=['stage_ft', 'vol_cu_ft', 'stage_m', 'total_vol_cu_m'])
# print stage_vol_df

stage_vol_df.drop(['stage_ft', 'vol_cu_ft'], inplace=True, axis=1)
stage_vol_df.set_index(stage_vol_df['stage_m'], inplace=True)
stage_vol_df.sort_index(inplace=True)

# fig = plt.figure()
# plt.plot(stage_vol_df['stage_m'], stage_vol_df['total_vol_cu_m'], 'r-o')
# plt.show()

water_balance_df = weather_df[['Evaporation (mm/day)', 'stage(m)', 'mean_stage_m']]
water_balance_df.loc[:, 'volume (cu.m)'] = 0.000


for index, row in water_balance_df.iterrows():
    obs_stage = row['stage(m)']  # observed stage
    if obs_stage >= stage_cutoff:
        x1, x2 = cd.find_range(stage_vol_df['stage_m'].tolist(), obs_stage)
        x_diff = x2 - x1
        y1 = stage_vol_df.loc[x1, 'total_vol_cu_m']
        y2 = stage_vol_df.loc[x2, 'total_vol_cu_m']
        y_diff = y2 - y1
        slope = y_diff / x_diff
        y_intercept = y2 - (slope * x2)
        water_balance_df.loc[index.strftime(daily_format), 'volume (cu.m)'] = (slope * obs_stage) + y_intercept

"""
full volume calculation
"""
x1, x2 = cd.find_range(stage_vol_df['stage_m'].tolist(), full_stage)
x_diff = x2 - x1
y1 = stage_vol_df.loc[x1, 'total_vol_cu_m']
y2 = stage_vol_df.loc[x2, 'total_vol_cu_m']
y_diff = y2 - y1
slope = y_diff / x_diff
y_intercept = y2 - (slope * x2)
full_volume = (slope*full_stage) + y_intercept
# print("full volume = %s" % full_volume)
"""
Overflow
"""
water_balance_df.loc[:, 'overflow(cu.m)'] = 0.000
for index, row in water_balance_df.iterrows():
    obs_volume = row['volume (cu.m)']
    if obs_volume > full_volume:
        overflow_volume = obs_volume - full_volume
        water_balance_df.loc[index.strftime(date_format), 'overflow(cu.m)'] = overflow_volume

print water_balance_df['overflow(cu.m)'].sum()

"""
Stage vs area linear relationship
"""
stage_area_df = pd.read_csv('/media/kiruba/New Volume/milli_watershed/tmg_lake_bathymetry/stage_volume_area/stage_area_tmg.csv',
                            sep=',', header=0, names=['stage_ft', 'area_sq_ft', 'stage_m', 'total_area_sq_m'])
stage_area_df.drop(['stage_ft', 'area_sq_ft'], inplace=True, axis=1)
# set stage as index
stage_area_df.set_index(stage_area_df['stage_m'], inplace=True)
# create empty column
water_balance_df.loc[:, 'ws_area(sq.m)'] = 0.000
for index, row in water_balance_df.iterrows():
    obs_stage = row['stage(m)']  # observed stage
    if obs_stage >= stage_cutoff:
        x1, x2 = cd.find_range(stage_area_df['stage_m'].tolist(), obs_stage)
        x_diff = x2 - x1
        y1 = stage_area_df.loc[x1, 'total_area_sq_m']
        y2 = stage_area_df.loc[x2, 'total_area_sq_m']
        y_diff = y2 - y1
        slope = y_diff / x_diff
        y_intercept = y2 - (slope * x2)
        water_balance_df.loc[index.strftime(date_format), 'ws_area(sq.m)'] = (slope * obs_stage) + y_intercept

"""
Evaporation Volume estimation
"""
water_balance_df.loc[:, 'Evaporation (cu.m)'] = (water_balance_df.loc[:, 'Evaporation (mm/day)'] * 0.001) * water_balance_df.loc[:, 'ws_area(sq.m)']
"""
Change in storage
"""
min_date = min(water_balance_df.index)
water_balance_df.loc[:, 'change_storage (cu.m)'] = 0.00
for index in water_balance_df.index:
    if index > min_date:
        previous_date = index - timedelta(days=1)
        d1_storage = water_balance_df.loc[previous_date.strftime(daily_format), 'volume (cu.m)']
        d2_storage = water_balance_df.loc[index.strftime(daily_format), 'volume (cu.m)']
        water_balance_df.loc[index.strftime(daily_format), 'change_storage (cu.m)'] = d2_storage - d1_storage 
        
print "change_Storage"
"""
Separate out no inflow/ non rainy days
two continuous days of no rain
"""
water_balance_df.loc[:, 'status'] = "Y"
min_date = min(water_balance_df.index) + timedelta(days=1)
for index in water_balance_df.index:
    if index > min_date and (water_balance_df.loc[index.strftime(daily_format), "change_storage (cu.m)"] < 0) and (water_balance_df.loc[index.strftime(daily_format), 'overflow(cu.m)'] == 0) and (abs(water_balance_df.loc[index.strftime(daily_format), 'change_storage (cu.m)']) > water_balance_df.loc[index.strftime(daily_format), 'Evaporation (cu.m)']):
        water_balance_df.loc[index.strftime(daily_format), 'status'] = "N"

dry_water_balance_df = water_balance_df[water_balance_df['status'] == "N"]
rain_water_balance_df = water_balance_df[water_balance_df['status'] == "Y"]
print "dry day sep"
print dry_water_balance_df.head()
print rain_water_balance_df.head()

"""
Calculate infiltration
"""
dry_water_balance_df.loc[:, 'infiltration (cu.m)'] = 0.00
delta_s = water_balance_df.loc[:, 'change_storage (cu.m)']
evap = water_balance_df.loc[:, 'Evaporation (cu.m)']
outflow = water_balance_df.loc[:, 'overflow(cu.m)']
min_date = min(water_balance_df.index)
max_date = max(water_balance_df.index)
for index, row in dry_water_balance_df.iterrows():
    if index > min_date:
        infilt = (-1.0 * (dry_water_balance_df.loc[index, 'change_storage (cu.m)'] + dry_water_balance_df.loc[index, 'Evaporation (cu.m)']))
        dry_water_balance_df.loc[index.strftime(daily_format), 'infiltration (cu.m)'] = infilt
# dry_water_balance_df[:, 'infiltration(cu.m)'] = cd.myround(dry_water_balance_df['infiltration(cu.m)'], decimals=3)
# dry_water_balance_df = dry_water_balance_df[dry_water_balance_df['stage(m)'] > 0.1]
# dry_water_balance_df = dry_water_balance_df[dry_water_balance_df['infiltration (cu.m)'] > 1.0]
# dry_water_balance_df = dry_water_balance_df[dry_water_balance_df['infiltration(cu.m)'] < 60]
dry_water_balance_df['infiltration_rate (m)'] = dry_water_balance_df['infiltration (cu.m)']/dry_water_balance_df['ws_area(sq.m)']
average_infiltration_rate = cd.myround(dry_water_balance_df['infiltration_rate (m)'].mean(), decimals=3)
print "infilt"
"""
Rainy day infiltration
"""
rain_water_balance_df.loc[:, 'infiltration (cu.m)'] = 0.0
for i in rain_water_balance_df.index:
    if rain_water_balance_df.loc[i.strftime(daily_format), 'stage(m)'] >= stage_cutoff:
        surface_area = rain_water_balance_df['ws_area(sq.m)'][i.strftime(daily_format)]
        rain_water_balance_df['infiltration (cu.m)'][i.strftime(daily_format)] = average_infiltration_rate*surface_area
print "rainy day"
"""
Inflow calculation
"""
merged_water_balance = pd.concat([dry_water_balance_df, rain_water_balance_df])
merged_water_balance.sort_index(inplace=True)
merged_water_balance.loc[:, 'Inflow (cu.m)'] = 0.000
delta_s_rain = merged_water_balance['change_storage (cu.m)']
inf_rain = merged_water_balance['infiltration (cu.m)']
evap_rain = merged_water_balance['Evaporation (cu.m)']
outflow_rain = merged_water_balance['overflow(cu.m)']
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
      merged_water_balance['infiltration (cu.m)'].sum() +
      merged_water_balance['overflow(cu.m)'].sum()) - merged_water_balance['Inflow (cu.m)'].sum()

print "E =", merged_water_balance['Evaporation (cu.m)'].sum()
print "Infil=", merged_water_balance['infiltration (cu.m)'].sum()
print "Overflow=", merged_water_balance['overflow(cu.m)'].sum()
print "Inflow =", merged_water_balance['Inflow (cu.m)'].sum()
print "Storage=", wb

merged_water_balance.index.name = 'Date'
# merged_water_balance['cum_rain'] = merged_water_balance['rain (mm)'].cumsum()
print merged_water_balance.dtypes
# raise SystemExit(0)
merged_water_balance['Inflow (cu.m)'] = merged_water_balance['Inflow (cu.m)'].astype(float)
merged_water_balance.to_csv('/media/kiruba/New Volume/milli_watershed/tmg_lake_water_balance/tmg_daily_wb.csv')
print merged_water_balance.head()
# print water_balance_df.head()