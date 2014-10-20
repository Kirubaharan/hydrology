__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from matplotlib import rc
from spread import spread
import meteolib as met
import evaplib
from bisect import bisect_left, bisect_right
from scipy.optimize import curve_fit
import math
import scipy as sp
from datetime import timedelta
import matplotlib as mpl

# latex parameters
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)

# Weather file
weather_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/corrected_weather.csv'
# Rain file
rain_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/corrected_rain_data.csv'

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

rain_df['Date_Time'] = pd.to_datetime(rain_df['Date_Time'],format=date_format)
rain_df.set_index(rain_df['Date_Time'], inplace=True)
# sort based on index
rain_df.sort_index(inplace=True)
# drop date time column
rain_df = rain_df.drop('Date_Time', 1)
# print rain_df.head()
# rain_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/rain_corr.csv')
# create daily value
rain_daily_df = rain_df.resample('D', how=np.sum)
rain_daily_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/rain_daily_corr.csv')
# create average daily weather
weather_daily_df = weather_df.resample('D', how=np.mean)
weather_daily_df = weather_daily_df.join(rain_daily_df, how='left')
"""
Evaporation from open water
Equation according to J.D. Valiantzas (2006). Simplified versions
for the Penman evaporation equation using routine weather data.
J. Hydrology 331: 690-702. Following Penman (1948,1956). Albedo set
at 0.06 for open water.
Input (measured at 2 m height):
        - airtemp: (array of) daily average air temperatures [Celsius]
        - rh: (array of) daily average relative humidity [%]
        - airpress: (array of) daily average air pressure data [Pa]
        - Rs: (array of) daily incoming solar radiation [J/m2/day]
        - N: (array of) maximum daily sunshine hours [h]
        - Rext: (array of) daily extraterrestrial radiation [J/m2/day]
        - u: (array of) daily average wind speed at 2 m [m/s]
        - Z: (array of) site elevation [m a.s.l.], default is zero...

    Output:
        - E0: (array of) Penman open water evaporation values [mm/day]

"""
"""
 air pressure (Pa) = 101325(1-2.25577 10^-5 h)^5.25588
h = altitude above sea level (m)
http://www.engineeringtoolbox.com/air-altitude-pressure-d_462.html
mean elevation over watershed = 803.441589 m
Elevation at the check dam = 799 m
"""
z = 799
p = (1-(2.25577*(10**-5)*z))
air_p_pa = 101325*(p**5.25588)
  # give air pressure value
weather_daily_df['AirPr(Pa)'] = air_p_pa
"""
Sunshine hours calculation
591 calculation - lat = 13.260196, long = 77.5120849
floor division(//): http://www.tutorialspoint.com/python/python_basic_operators.htm
"""
#select radiation data separately
sunshine_df = weather_df[['Solar Radiation (W/mm2)']]
# give value 1 if there is radiation and rest 0
sunshine_df['sunshine hours (h)'] = (sunshine_df['Solar Radiation (W/mm2)'] != 0).astype(int)  # gives 1 for true and 0 for false
# print sunshine_df.head()
#aggregate the values to daily and divide it by 2 to get sunshine hourly values
sunshine_daily_df = sunshine_df.resample('D', how=np.sum) // 2  # // floor division
sunshine_daily_df = sunshine_daily_df.drop(sunshine_daily_df.columns.values[0], 1)
# print sunshine_daily_df.head()
weather_daily_df = weather_daily_df.join(sunshine_daily_df, how='left')
"""
Daily Extraterrestrial Radiation Calculation(J/m2/day)
"""


def rext_calc(df, lat=float):
    """
    Function to calculate extraterrestrial radiation output in J/m2/day
    Ref:http://www.fao.org/docrep/x0490e/x0490e07.htm

    :param df: dataframe with datetime index
    :param lat: latitude (negative for Southern hemisphere)
    :return: Rext (J/m2)
    """
    # set solar constant [MJ m^-2 min^-1]
    s = 0.08166
    #convert latitude [degrees] to radians
    latrad = lat*math.pi / 180.0
    #have to add in function for calculating single value here
    # extract date, month, year from index
    date = pd.DatetimeIndex(df.index).day
    month = pd.DatetimeIndex(df.index).month
    year = pd.DatetimeIndex(df.index).year
    doy = met.date2doy(dd=date, mm=month, yyyy=year)  # create day of year(1-366) acc to date
    l = sp.size(doy)
    if l < 2:
        dt = 0.409 * math.sin(2 * math.pi / 365 * doy - 1.39)
        ws = sp.arccos(-math.tan(latrad) * math.tan(dt))
        j = 2 * math.pi / 365.25 * doy
        dr = 1.0 + 0.03344 * math.cos(j - 0.048869)
        rext = s * 1440 / math.pi * dr * (ws * math.sin(latrad) * math.sin(dt) + math.sin(ws) * math.cos(latrad) * math.cos(dt))
    #Create dummy output arrays sp refers to scipy
    else:
        rext = sp.zeros(l)
        dt = sp.zeros(l)
        ws = sp.zeros(l)
        j = sp.zeros(l)
        dr = sp.zeros(l)
        #calculate Rext
        for i in range(0, l):
            #Calculate solar decimation dt(d in FAO) [rad]
            dt[i] = 0.409 * math.sin(2 * math.pi / 365 * doy[i] - 1.39)
            #calculate sunset hour angle [rad]
            ws[i] = sp.arccos(-math.tan(latrad) * math.tan(dt[i]))
            # calculate day angle j [radians]
            j[i] = 2 * math.pi / 365.25 * doy[i]
            # calculate relative distance to sun
            dr[i] = 1.0 + 0.03344 * math.cos(j[i] - 0.048869)
            #calculate Rext dt = d(FAO) and latrad = j(FAO)
            rext[i] = (s * 1440.0 / math.pi) * dr[i] * (ws[i] * math.sin(latrad) * math.sin(dt[i]) + math.sin(ws[i])* math.cos(latrad) * math.cos(dt[i]))

    rext = sp.array(rext) * 1000000
    return rext

weather_daily_df['Rext (J/m2)'] = rext_calc(weather_daily_df, lat=13.260196)
"""
wind speed from km/h to m/s
1 kmph = 0.277778 m/s
"""
weather_daily_df['Wind Speed (mps)'] = weather_daily_df['Wind Speed (kmph)'] * 0.277778
"""
Radiation unit conversion
"""
weather_daily_df['Solar Radiation (J/m2/day)'] = weather_daily_df['Solar Radiation (W/mm2)'] * 86400
"""
Pot Evaporation calculation
"""
airtemp = weather_daily_df['Air Temperature (C)']
hum = weather_daily_df['Humidity (%)']
airpress = weather_daily_df['AirPr(Pa)']
rs = weather_daily_df['Solar Radiation (J/m2/day)']
sun_hr = weather_daily_df['sunshine hours (h)']
rext = weather_daily_df['Rext (J/m2)']
wind_speed = weather_daily_df['Wind Speed (mps)']
weather_daily_df['Evaporation (mm/day)'] = evaplib.E0(airtemp=airtemp, rh=hum, airpress=airpress, Rs=rs, N=sun_hr, Rext=rext, u=wind_speed, Z=z )
"""
Plot Evaporation
"""
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_daily_df.index, weather_daily_df['Evaporation (mm/day)'], '-g', label='Evaporation (mm/day)')
plt.ylabel(r'\textbf{Evaporation ($mm/day$)}')
fig.autofmt_xdate(rotation=90)
plt.title(r"Daily Evaporation for Check Dam - 591", fontsize=20)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_591/evaporation_591')
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
water_level_1 = water_level_1.resample('D', how=np.mean)
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
water_level_2 = water_level_2.resample('D', how=np.mean)
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
water_level_3 = water_level_3.resample('D', how=np.mean)
# print water_level_3
# bloack 4
block_4 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525_008_004.CSV'
water_level_4 = pd.read_csv(block_4, skiprows=9, sep=',', header=0,  names=['scan no', 'date',
                                                                            'time', 'raw value', 'calibrated value'])
water_level_4['calibrated value'] = (water_level_4['raw value']*coeff_cal[0]) + coeff_cal[1]  # in cm
# convert to metre
water_level_4['calibrated value'] /= 100
#change the column name
water_level_4.columns.values[4] = 'stage(m)'
# create date time index
format = '%d/%m/%Y  %H:%M:%S'
# change 24:00:00 to 23:59:59
water_level_4['time'] = water_level_4['time'].replace(' 24:00:00', ' 23:59:59')
water_level_4['date_time'] = pd.to_datetime(water_level_4['date'] + water_level_4['time'], format=format)
water_level_4.set_index(water_level_4['date_time'], inplace=True)
# drop unneccessary columns before datetime aggregation
water_level_4.drop(['scan no', 'date', 'time', 'raw value'], inplace=True, axis=1)
#aggregate daily
water_level_4 = water_level_4.resample('D', how=np.mean)
water_level = pd.concat([water_level_1, water_level_2, water_level_3, water_level_4], axis=0)

weather_daily_df = weather_daily_df.join(water_level, how='left')

"""
Remove Duplicates
"""
# check for duplicates
# df2 = dry_weather.groupby(level=0).filter(lambda x: len(x) > 1)
# print(df2)
weather_daily_df['index'] = weather_daily_df.index
weather_daily_df.drop_duplicates(subset='index', take_last=True, inplace=True)
del weather_daily_df['index']
weather_daily_df = weather_daily_df.sort()
"""
Stage Volume relation estimation from survey data
"""
# neccessary functions


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2,s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)
#function to create stage volume output


def calcvolume(profile, order, dy):
    """
    Profile = df.Y1,df.Y2,.. and order = 1,2,3
    :param profile: series of Z values
    :param order: distance from origin
    :param dy: thickness of profile in m
    :param dam_height: Height of check dam in m
    :return: output: pandas dataframe volume for profile
    """

    # print 'profile length = %s' % len(profile)
    results = []

    for stage in dz:
        water_area = 0
        for z1, z2 in pairwise(profile):
            delev = (z2 - z1) / 10
            elev = z1
            for b in range(1, 11, 1):
                elev += delev
                if stage > elev:
                    # print 'elev = %s' % elev
                    water_area += (0.1 * (stage-elev))
                    # print 'order = %s and dy = %s' %(order, dy)
                    # print 'area = %s' % water_area

            calc_vol = water_area * dy
        # print 'calc vol = %s' % calc_vol
        results.append(calc_vol)
        # print 'results = %s' % results
        # print 'results length = %s' % len(results)

    output[('Volume_%s' % order)] = results
#input parameters
base_file_591 = '/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/base_profile_591.csv'
check_dam_no = 591
check_dam_height = 1.9    # m
df_591 = pd.read_csv(base_file_591, sep=',')
df_591_trans = df_591.T  # Transpose
no_of_stage_interval = check_dam_height/.05
dz = list((spread(0.00, check_dam_height, int(no_of_stage_interval), mode=3)))
index = [range(len(dz))]  # no of stage intervals
columns = ['stage_m']
data = np.array(dz)
output = pd.DataFrame(data, index=index, columns=columns)
# print(df_591_trans)
# print len(df_591_trans.ix[1:, 0])
### Renaming the column and dropping y values
y_name_list = []
for y_value in df_591_trans.ix[0, 0:]:
    y_name_list.append(('Y_%d' %y_value))

df_591_trans.columns = y_name_list
# print df_591_trans
y_value_list = df_591_trans.ix[0, 0:]
# print y_value_list

# drop the y values from data
final_data = df_591_trans.ix[1:, 0:]
# print final_data

#volume calculation
for l1, l2 in pairwise(y_value_list):
    calcvolume(profile=final_data["Y_%d" % l1], order=l1, dy=int(l2-l1))

output_series = output.filter(regex="Volume_")  # filter the columns that have Volume_
output["total_vol_cu_m"] = output_series.sum(axis=1)  # get total volume
# print output

# select only stage and total volume
stage_vol_df = output[['stage_m', "total_vol_cu_m"]]
"""
Select data where stage is available, Remove Overflowing days
"""
weather_stage_avl_df = weather_daily_df[min(water_level.index):max(water_level.index)]

# weather_stage_avl_df = weather_stage_avl_df[weather_stage_avl_df['stage(m)'] < 1.9]
# assumption cutoff stage to be 14 cm below which data is not considered reliable
# weather_stage_avl_df = weather_stage_avl_df[weather_stage_avl_df['stage(m)'] > 0.05]
# weather_stage_avl_df = weather_stage_avl_df[weather_stage_avl_df['change_storage(cu.m)'] > 0]
# print weather_stage_avl_df['stage(m)']
"""
Convert observed stage to volume by linear interpolation
"""
# set stage as index
stage_vol_df.set_index(stage_vol_df['stage_m'], inplace=True)
# function to find containing intervals


def find_range(array, ab):
    if ab < max(array):
        start = bisect_left(array, ab)
        return array[start-1], array[start]
    else:
        return min(array), max(array)



# print weather_stage_avl_df.head()
water_balance_df = weather_stage_avl_df[['Rain Collection (mm)', 'Evaporation (mm/day)', 'stage(m)']]
# print find_range(stage_vol_df['stage_m'].tolist(), max(water_balance_df['stage(m)']))
water_balance_df['volume (cu.m)'] = 0.000
for index, row in water_balance_df.iterrows():
    obs_stage = row['stage(m)']  # observed stage
    x1, x2 = find_range(stage_vol_df['stage_m'].tolist(), obs_stage)
    x_diff = x2-x1
    y1 = stage_vol_df['total_vol_cu_m'][x1]
    y2 = stage_vol_df['total_vol_cu_m'][x2]
    y_diff = y2 - y1
    slope = y_diff/x_diff
    y_intercept = y2 - (slope*x2)
    water_balance_df['volume (cu.m)'][index.strftime('%Y-%m-%d')] = (slope*obs_stage) + y_intercept

# fig = plt.figure(figsize=(11.69, 8.27))
# plt.plot_date(water_balance_df.index, water_balance_df['volume (cu.m)'], '-g')
# plt.hlines(stage_vol_df['total_vol_cu_m'][1.9], min(water_balance_df.index), max(water_balance_df.index))
# plt.title('before overflow correction')

"""
Overflow
"""
full_vol = stage_vol_df['total_vol_cu_m'][1.9]
print full_vol
water_balance_df['overflow(cu.m)'] = 0.000
for index, row in water_balance_df.iterrows():
    obs_vol = row['volume (cu.m)']
    if obs_vol > full_vol:
        # print obs_vol
        water_balance_df['overflow(cu.m)'][index.strftime('%Y-%m-%d')] = obs_vol - full_vol
        water_balance_df['volume (cu.m)'][index.strftime('%Y-%m-%d')] = full_vol
        water_balance_df['stage(m)'][index.strftime('%Y-%m-%d')] = 1.9


# fig = plt.figure(figsize=(11.69, 8.27))
# plt.plot_date(water_balance_df.index, water_balance_df['volume (cu.m)'], '-g')
# plt.hlines(stage_vol_df['total_vol_cu_m'][1.9], min(water_balance_df.index), max(water_balance_df.index))
# plt.title('after overflow correction')
"""
Stage vs area linear relationship
"""
stage_area_df = pd.read_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/cont_area.csv',
                            sep=',', header=0, names=['sno', 'stage_m', 'total_area_sq_m'])
stage_area_df.drop('sno', inplace=True, axis=1)
# set stage as index
stage_area_df.set_index(stage_area_df['stage_m'], inplace=True)
# print max(water_balance_df['stage(m)'])
# print find_range(stage_area_df['stage_m'].tolist(), max(water_balance_df['stage(m)']))
#create empty column
water_balance_df['ws_area(sq.m)'] = 0.000
for index, row in water_balance_df.iterrows():
    obs_stage = row['stage(m)']  # observed stage
    x1, x2 = find_range(stage_area_df['stage_m'].tolist(), obs_stage)
    x_diff = x2-x1
    y1 = stage_area_df['total_area_sq_m'][x1]
    y2 = stage_area_df['total_area_sq_m'][x2]
    y_diff = y2 - y1
    slope = y_diff/x_diff
    y_intercept = y2 - (slope*x2)
    water_balance_df['ws_area(sq.m)'][index.strftime('%Y-%m-%d')] = (slope*obs_stage) + y_intercept
"""
Evaporation Volume estimation
"""
water_balance_df['Evaporation (cu.m)'] = (water_balance_df['Evaporation (mm/day)'] * 0.001) * water_balance_df['ws_area(sq.m)']
"""
change in storage
"""
#assume 0 initially
water_balance_df['change_storage(cu.m)'] = 0.000

#change in storage is today minus yesterday volume
for d1, d2 in pairwise(water_balance_df.index):
    if d2 > d1:
        diff = (d2-d1).days
        if diff == 1:
            water_balance_df['change_storage(cu.m)'][d2.strftime('%Y-%m-%d')] = water_balance_df['volume (cu.m)'][d2.strftime('%Y-%m-%d')] - water_balance_df['volume (cu.m)'][d1.strftime('%Y-%m-%d')]
# print water_balance_df
# water_balance_df = water_balance_df[water_balance_df['change_storage(cu.m)'] < 0]


def plot_date(dataframe, column_name):
    """

    :param dataframe:
    :param column_name:
    :type column_name:str
    :return:
    """
    fig = plt.figure(figsize=(11.69, 8.27))
    p = plt.plot(dataframe.index, dataframe[column_name], 'b-', label=r"%s" % column_name)
    plt.hlines(0, min(dataframe.index), max(dataframe.index), 'r')
    plt.legend(loc='best')
    fig.autofmt_xdate(rotation=90)
    return p

a = plot_date(water_balance_df, 'change_storage(cu.m)')


# plt.show()
#create average stage for two days
water_balance_df['average_stage_m'] = 0.000
for d1, d2 in pairwise(water_balance_df.index):
    diff = abs((d2-d1).days)
    if diff == 1:
        water_balance_df['average_stage_m'][d2.strftime('%Y-%m-%d')] = (water_balance_df['stage(m)']
                                                                        [d2.strftime('%Y-%m-%d')]
                                                                        + water_balance_df['stage(m)']
                                                                        [d1.strftime('%Y-%m-%d')])/2
# print water_balance_df.head()
# print water_balance_df

"""
Separate inflow and no inflow days
"""
dry_water_balance_df = water_balance_df[water_balance_df['change_storage(cu.m)'] < 0]
rain_water_balance_df = water_balance_df[water_balance_df['change_storage(cu.m)'] > 0]

# b = plot_date(dry_water_balance_df, 'change_storage(cu.m)')
"""
Calculate infiltration
"""
# calculate infiltration
dry_water_balance_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/dry_wb_check.CSV')
# print dry_water_balance_df.head()
# dry_water_balance_df['infiltration(cu.m)'] = 0.000
delta_s = dry_water_balance_df['change_storage(cu.m)']
evap = dry_water_balance_df['Evaporation (cu.m)']
outflow = dry_water_balance_df['overflow(cu.m)']
# for t1, t2 in pairwise(dry_water_balance_df.index):
#     diff = abs((t2-t1).days)
#     if diff == 1:
#         print t1, t2
#         dry_water_balance_df['infiltration(cu.m)'][t1.strftime('%Y-%m-%d')] = -1*(delta_s[t2.strftime('%Y-%m-%d')] + evap[t2.strftime('%Y-%m-%d')] + outflow[t2.strftime('%Y-%m-%d')])
# # for index, row in dry_water_balance_df.iterrows():
#     if index > min(dry_water_balance_df.index):
#         t_1 = index - timedelta(days=1)
#         if t_1 < max(dry_water_balance_df.index):
#             diff = abs((index-t_1).days)
#             if diff == 1:
#                 print index
#                 # print t_1
                # dry_water_balance_df['infiltration(cu.m)'][index.strftime('%Y-%m-%d')] = -1*(delta_s[index.strftime('%Y-%m-%d')] + evap[t_1.strftime('%Y-%m-%d')] + outflow[t_1.strftime('%Y-%m-%d')])
    # print row


dry_water_balance_df['infiltration(cu.m)'] = -1.0*(evap + outflow + delta_s)
# print dry_water_balance_df.head()
# fig = plt.figure(figsize=(11.69, 8.27))
# plt.plot(dry_water_balance_df['average_stage_m'], dry_water_balance_df['infiltration(cu.m)'], 'bo')
# plt.show()
"""
Dry infiltration vs rainfall
"""
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(11.69, 8.27))
# fig.subplots_adjust(right=0.8)
line1 = ax1.plot(water_balance_df.index, water_balance_df['Rain Collection (mm)'], '-b', label=r'Rainfall(mm)')
plt.gca().invert_yaxis()
ax1.xaxis.tick_bottom()
ax1.yaxis.tick_left()
for t1 in ax1.get_yticklabels():
    t1.set_color('b')
# plt.legend(loc='upper left')
ax2 = ax1.twinx()
cmap, norm = mpl.colors.from_levels_and_colors([0, 0.05, 1, 1.5, 2.0], ['red', 'yellow', 'green', 'blue'])
line2 = ax2.scatter(dry_water_balance_df.index, dry_water_balance_df['infiltration(cu.m)'], label='Infiltration (cu.m)', c=dry_water_balance_df['stage(m)'], cmap=cmap, norm=norm)
plt.hlines(0, min(dry_water_balance_df.index), max(dry_water_balance_df.index))
ax2.xaxis.tick_bottom()
ax2.yaxis.tick_right()
for t1 in ax2.get_yticklabels():
    t1.set_color('r')
# plt.legend(loc='upper right')
# fig.autofmt_xdate(rotation=90)
# fig.subplots_adjust(right=0.8)
ax3 = ax2.twiny()
line3 = ax3.plot(water_balance_df.index, water_balance_df['Evaporation (cu.m)'], '-g', label='Evaporation (cu.m)' )
ax3.tick_params(axis='x',
                which='both',
                top='off',
                bottom='off',
                labeltop='off')
# ax3.xaxis.tick_bottom()
ax3.yaxis.tick_right()
fig.autofmt_xdate(rotation=90)
lns = line1+line3
labs = [l.get_label() for l in lns]
ax3.legend(lns, labs, loc='upper center', fancybox=True, ncol=3, bbox_to_anchor=(0.5, 1.15))
# ax3.set_xlim([min(dry_water_balance_df.index), max(dry_water_balance_df.index)])
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.50, 0.05, 0.3])  #first one distance from plot, second height
# cax, kw = mpl.colorbar.make_axes([ax for ax in ax1.flat()])
cbar = fig.colorbar(line2, cax=cbar_ax)
cbar.ax.set_ylabel('Stage (m)')
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_591/dry_rain_infiltration_stage_591')
plt.show()
"""
Fitting exponential function
"""
stage_cal = dry_water_balance_df['stage(m)']
# stage_cal = dry_water_balance_df['average_stage_m']
inf_cal = dry_water_balance_df['infiltration(cu.m)']


def func(h, alpha, beta):
    return alpha*(h**beta)

popt, pcov = curve_fit(func, stage_cal, inf_cal)

print popt
print pcov
# print np.diag(pcov)
print np.sqrt(np.diag(pcov))

# plot
stage_cal_new = np.linspace(min(stage_cal), max(stage_cal), 50)
inf_cal_new = func(stage_cal_new, *popt)
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(stage_cal, inf_cal, 'bo', label=r'Observation')
plt.plot(stage_cal_new, inf_cal_new, 'r-', label='Prediction')
plt.vlines(1.9, 0, max(inf_cal), 'g')
plt.hlines(0, min(stage_cal), max(stage_cal), 'y')
plt.legend(loc='upper left')
plt.xlabel(r'\textbf{Stage} (m)')
plt.ylabel(r'\textbf{Infiltration} ($m^3/day$)')
plt.title(r"No inflow day's stage - infiltration relationship for 591 check dam")
plt.text(x=0.15, y=1250, fontsize=15, s=r'$Infiltration = {0:.2f}h^{{{1:.2f}}}$'.format(popt[0], popt[1]))
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_591/stage_inf_exp_dry_591')
# plt.show()
# print dry_water_balance_df
# print dry_water_balance_df[dry_water_balance_df['infiltration(cu.m)'] < 0]
# plot rainfall vs stage

fig, ax1 = plt.subplots(figsize=(11.69, 8.27))
ax1.bar(water_balance_df.index, water_balance_df['Rain Collection (mm)'], 0.35, color='b', label=r'Rainfall(mm)')
plt.gca().invert_yaxis()
for t1 in ax1.get_yticklabels():
    t1.set_color('b')
ax1.set_ylabel('Rainfall(mm)')
plt.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.plot_date(water_balance_df.index, water_balance_df['stage(m)'], 'r', label='stage (m)')
for t1 in ax2.get_yticklabels():
    t1.set_color('r')
plt.legend(loc='upper right')
fig.autofmt_xdate(rotation=90)
# plt.show()

"""
Rainy day infiltration
"""
rain_water_balance_df['infiltration(cu.m)'] = popt[0]*(rain_water_balance_df['average_stage_m']**popt[1])
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(rain_water_balance_df['average_stage_m'], rain_water_balance_df['infiltration(cu.m)'], 'bo', label='Predicted Infiltration' )
# plt.vlines(1.9, 0, 100, 'g')
# plt.xlim([-1, 2.0])
# plt.legend(loc='upper left')
plt.xlabel(r'\textbf{Stage} (m)')
plt.ylabel(r'\textbf{Infiltration} ($m^3/day$)')
plt.title(r"Inflow day's stage - infiltration relationship for 591 check dam")
# plt.show()

"""
Inflow calculation
"""
# print dry_water_balance_df.head()
dry_water_balance_df['status'] = 'D'
rain_water_balance_df['status'] = 'R'
# dry_water_balance_df = dry_water_balance_df.drop(['Evaporation (mm/day)', 'ws_area(sq.m)'], inplace=True, axis=1)
# rain_water_balance_df = rain_water_balance_df.drop(['Evaporation (mm/day)', 'ws_area(sq.m)'], inplace=True, axis=1)
# merged_table = dry_water_balance_df.join(rain_water_balance_df, how='right')
# print rain_water_balance_df.head()
dry_water_balance_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/dry_wb.CSV')
rain_water_balance_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/rain_wb.CSV')