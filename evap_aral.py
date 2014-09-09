__author__ = 'kiruba'
"""
Calculate daily evapotranspiration from weather data.
and water balance for Check dam no 591
"""
import evaplib
import meteolib as met
import scipy as sp
import pandas as pd
# import gdal, ogr, osr, numpy, sys   # uncomment it if you want to use zonalstats
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib import rc
from datetime import date
import itertools
from spread import spread
# copy the code from http://code.activestate.com/recipes/577878-generate-equally-spaced-floats/ #
from bisect import bisect_left
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA



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
# print df_base.head()
# print df_base.columns.values[16]


"""
Aggregate half hourly to daily
"""

## separate out rain daily sum
rain_df = df_base[['Date_Time', 'Rain Collection (mm)']]
## separate out weather data(except rain)

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

# print weather_df.head()
#  Aggregate sum to get daily rainfall
# http://stackoverflow.com/questions/17001389/pandas-resample-documentation
rain_df = rain_df.resample('D', how=np.sum)   # D for day

# Aggregate average to get daily weather parameters
weather_daily_df = weather_df.resample('D', how=np.mean)

# print weather_daily_df.head()
#plot for daily rainfall
fig = plt.figure(figsize=(11.69, 8.27))
plt.bar(rain_df.index, rain_df['Rain Collection (mm)'], 0.35, color='b')
# plt.plot_date(rain_df.index, rain_df['Rain Collection (mm)'], 'b-')
fig.autofmt_xdate()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.xlabel(r'\textbf{Area} ($m^2$)')
plt.ylabel(r'\textbf{Rain} (mm)')
plt.title(r'Rainfall - Aralumallige Watershed')
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/rainfall_591')

# plt.show()

# print rain_df.head()
"""
Separate out dry and rainy days
"""
dry_days = rain_df[rain_df['Rain Collection (mm)'] == 0]
rain_days = rain_df[rain_df["Rain Collection (mm)"] > 0]
dry_dates = dry_days.index
rain_dates = rain_days.index
no_dry_days = len(dry_days)
no_rain_days = len(rain_days)
# rain = rain_df[]
# print rain['Rain Collection (mm)']
# print(no_dry_days,no_rain_days)
# print dry_dates
# print rain_dates
# fig = plt.figure(figsize=(11.69, 8.27))
# plt.subplot(1,1,1)
# plt.plot_date(dry_days.index, dry_days[[0]], color='red', linestyle='--')
# plt.plot_date(rain_days.index, rain_days[[0]], color='green', linestyle='--')
# plt.show()
"""
Merge weather data and rain data based on dry/rain day
"""
rain_weather = weather_daily_df.join(rain_days, how='right')  # right = use the index of rain days
# print rain_weather.head()
dry_weather = weather_daily_df.join(dry_days, how='right')
# print dry_weather.head()
# print df_base['2014-05-20']['Rain Collection (mm)']

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
dry_weather['AirPr(Pa)'] = air_p_pa
# print(dry_weather.head())


"""
Sunshine hours calculation
591 calculation - lat = 13.260196, long = 77.5120849
floor division(//): http://www.tutorialspoint.com/python/python_basic_operators.htm
"""
#select radiation data separately
sunshine_df = weather_df[['Date_Time', 'Solar Radiation (W/mm2)']]
# give value 1 if there is radiation and rest 0
sunshine_df['sunshine hours (h)'] = (sunshine_df['Solar Radiation (W/mm2)'] != 0).astype(int)  # gives 1 for true and 0 for false
# print sunshine_df.head()
#aggregate the values to daily and divide it by 2 to get sunshine hourly values
sunshine_daily_df = sunshine_df.resample('D', how=np.sum) // 2  # // floor division
sunshine_daily_df = sunshine_daily_df.drop(sunshine_daily_df.columns.values[0], 1)
# Join with rainy day and dry day weather based on date
# rain_weather = rain_weather.join(sunshine_daily_df, how='left')
dry_weather = dry_weather.join(sunshine_daily_df, how='left')


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
    s = 0.0820
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
        rext = s * 86400 / math.pi * dr * (ws * math.sin(latrad) * math.sin(dt) + math.sin(ws) * math.cos(latrad) * math.cos(dt))
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
            rext[i] = (s * 86400.0 / math.pi) * dr[i] * (ws[i] * math.sin(latrad) * math.sin(dt[i]) + math.sin(ws[i])* math.cos(latrad) * math.cos(dt[i]))

    rext = sp.array(rext) * 1000000
    return rext

dry_weather['Rext (J/m2)'] = rext_calc(dry_weather, lat=13.260196)
# print dry_weather

# fig = plt.figure()
# plt.plot_date(dry_weather.index, dry_weather['Rext (J/m2)'], 'b*')
# fig.autofmt_xdate()
# plt.show()

"""
wind speed from km/h to m/s
1 kmph = 0.277778 m/s
"""
dry_weather['Wind Speed (mps)'] = dry_weather['Wind Speed (kmph)'] * 0.277778
# print rain_weather

"""
Radiation unit conversion
"""

dry_weather['Solar Radiation (J/m2/day)'] = dry_weather['Solar Radiation (W/mm2)'] * 86400

"""
Dry weather Evaporation calculation
"""
airtemp_d = dry_weather['Air Temperature (C)']
hum_d = dry_weather['Humidity (%)']
airpress_d = dry_weather['AirPr(Pa)']
rs_d = dry_weather['Solar Radiation (J/m2/day)']
sun_hr_d = dry_weather['sunshine hours (h)']
rext_d = dry_weather['Rext (J/m2)']
wind_speed_d = dry_weather['Wind Speed (mps)']


eo_d = evaplib.E0(airtemp=airtemp_d, rh=hum_d, airpress=airpress_d, Rs=rs_d, N=sun_hr_d, Rext=rext_d, u=wind_speed_d, Z=z )
dry_weather['Evaporation (mm/day)'] = eo_d



"""
Wet weather Evaporation calculation
"""
# air pressure
rain_weather['AirPr(Pa)'] = air_p_pa
# sunshine hours
rain_weather = rain_weather.join(sunshine_daily_df, how='left')
#extraterrestrial radiation
rain_weather['Rext (J/m2)'] = rext_calc(rain_weather, lat=13.260196)
# wind speed unit conversion
rain_weather['Wind Speed (mps)'] = rain_weather['Wind Speed (kmph)'] * 0.277778
#radiation unit conversion
rain_weather['Solar Radiation (J/m2/day)'] = rain_weather['Solar Radiation (W/mm2)'] * 86400
airtemp_r = rain_weather['Air Temperature (C)']
hum_r = rain_weather['Humidity (%)']
airpress_r = rain_weather['AirPr(Pa)']
rs_r = rain_weather['Solar Radiation (J/m2/day)']
sun_hr_r = rain_weather['sunshine hours (h)']
rext_r = rain_weather['Rext (J/m2)']
wind_speed_r = rain_weather['Wind Speed (mps)']
eo_r = evaplib.E0(airtemp=airtemp_r, rh=hum_r, airpress=airpress_r, Rs=rs_r, N=sun_hr_r, Rext=rext_r, u=wind_speed_r,Z=z)
rain_weather['Evaporation (mm/day)'] = eo_r
dry_weather.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/dry_evap_591.csv')
rain_weather.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/rain_evap_591.csv')
#plot
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(dry_weather.index, dry_weather['Evaporation (mm/day)'], 'r.', label='Non -Rainy Day')
plt.plot_date(rain_weather.index, rain_weather['Evaporation (mm/day)'], 'b.', label='Rainy Day')
fig.autofmt_xdate()
plt.legend(loc='upper right')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.ylabel(r'\textbf{Evaporation} ($mm/day$)')
plt.title(r"Daily Evaporation for Check Dam - 591", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/591_evap')
# plt.show()

"""
#plot weather parameters
print weather_daily_df.head()
#windspeed
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_daily_df.index, weather_daily_df['Wind Speed (kmph)'], 'r-', label='Wind Speed (Kmph)')
plt.ylabel(r'\textbf{Wind Speed}($Km/h$)')
plt.title(r"Wind Speed - Aralumallige", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/wind_speed_aral')
fig.autofmt_xdate()
#Air Temperature
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_daily_df.index, weather_daily_df['Air Temperature (C)'], 'r-', label='Air Temperature (C)')
plt.ylabel(r'\textbf{Air Temperature}(C)')
plt.title(r"Daily Average Temperature(C) - Aralumallige", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/avg_temp')
fig.autofmt_xdate()
#humidity
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_daily_df.index, weather_daily_df['Humidity (%)'], 'r-', label='Humidity (%)')
plt.ylabel(r'\textbf{Humidity}(\%)')
plt.title(r"Humidity - Aralumallige", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/humidity')
fig.autofmt_xdate()
#solar radiation
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_daily_df.index, weather_daily_df['Solar Radiation (W/mm2)'], 'r-', label='Solar Radiation (W/mm2)')
plt.ylabel(r'\textbf{Solar Radiation ($W/mm2$)')
plt.title(r"Solar Radiation - Aralumallige", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/solar_rad')
fig.autofmt_xdate()
plt.show()
"""
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
water_level = pd.concat([water_level_1, water_level_2, water_level_3], axis=0)
# print water_level
# merge with dry day weather and rain weather df
dry_weather = dry_weather.join(water_level, how='left')
rain_weather = rain_weather.join(water_level, how='left')
# select from the date where stage data is available
print min(dry_weather.index)
print max(dry_weather.index)
print min(rain_weather.index)
print max(rain_weather.index)
print min(water_level.index)
print max(water_level.index)
dry_weather = dry_weather[min(water_level.index):max(water_level.index)]
rain_weather = rain_weather[min(water_level.index):max(water_level.index)]
# print(rain_weather)
#save as csv
dry_weather.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/dry_stage_weather_591.csv')
rain_weather.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/rain_stage_weather_591.csv')
# plot daily water level
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(water_level.index, water_level['stage(m)'], 'r-', label='Stage (m)')
# plt.ylabel(r'\textbf{Wind Speed}($Km/h$)')
plt.title(r"Average Daily Water Level in Checkdam 591", fontsize=16)
plt.legend(loc='upper left')
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/daily_stage_591')
fig.autofmt_xdate()
# plt.show()

"""
# print stage_area_df
###calibration
z_cal = stage_area_df.ix[0:, 0]     # x
area_cal = stage_area_df.ix[0:, 1]       # y

stage_area_cal = polyfit(z_cal, area_cal, 2)    #
po_stage_area = np.polyfit(z_cal, area_cal, 2)
f_stage_area = np.poly1d(po_stage_area)
# print np.poly1d(f_stage_area)
# print stage_area_cal
# print a_stage['polynomial'][0]
coeff_stage_area_cal = stage_area_cal['polynomial']

#calculate new coefficients
z_cal_new = np.linspace(min(z_cal), max(z_cal), 50)
area_cal_new = ((z_cal_new**2)*coeff_stage_area_cal[0]) + \
               (z_cal_new*coeff_stage_area_cal[1]) + \
               coeff_stage_area_cal[2]
"""
# plt.show()

# Stage Volume relationship 591

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
Remove Duplicates
"""
# check for duplicates
# df2 = dry_weather.groupby(level=0).filter(lambda x: len(x) > 1)
# print(df2)
dry_weather['index'] = dry_weather.index
dry_weather.drop_duplicates(subset='index', take_last=True, inplace=True)
del dry_weather['index']# wet_water_balance = rain_weather[[]]
dry_weather = dry_weather.sort()
#  similar for rainy days
rain_weather['index'] = rain_weather.index
rain_weather.drop_duplicates(subset='index', take_last=True, inplace=True)
del rain_weather['index']
rain_weather = rain_weather.sort()

## Dry days water balance

# linear interpolation for stage volume
# set stage as index
stage_vol_df.set_index(stage_vol_df['stage_m'], inplace=True)
# print stage_vol_df
# function to find containing intervals


def find_range(array, a):
    start = bisect_left(array, a)
    return array[start-1], array[start]
# print dry_weather

dry_water_balance = dry_weather[['Rain Collection (mm)', 'Evaporation (mm/day)', 'stage(m)']]
# set stage as index
# dry_water_balance.set_index(dry_water_balance['stage(m)'], inplace=True)
# create empty volume column
dry_water_balance['volume (cu.m)'] = 0
# print dry_water_balance
# convert stage to volume
for index, row in dry_water_balance.iterrows():
    # print 'index = %s' % index
    # print 'row = %s' % row
    obs_stage = row['stage(m)']  # observed stage
    # print obs_stage
    x1, x2 = find_range(stage_vol_df['stage_m'].tolist(), obs_stage)
    x_diff = x2-x1
    y1 = stage_vol_df['total_vol_cu_m'][x1]
    y2 = stage_vol_df['total_vol_cu_m'][x2]
    y_diff = y2 - y1
    slope = y_diff/x_diff
    y_intercept = y2 - (slope*x2)
    # print x1, x2, stage_vol_df['total_vol_cu_m'][x1], stage_vol_df['total_vol_cu_m'][x2], slope
    dry_water_balance['volume (cu.m)'][index.strftime('%Y-%m-%d')] = (slope*obs_stage) + y_intercept
    # dry_water_balance['volume (cu.m)'][index.strftime('%Y-%m-%d')] = ((-1*x2) + obs_stage + (slope*y2))/slope

# dry_water_balance['volume (cu.m)'] = (coeff_stage_vol_cal[0]*(dry_water_balance['stage(m)']**2)) + \
#                                      (coeff_stage_vol_cal[1]*dry_water_balance['stage(m)']) + \
#                                      coeff_stage_vol_cal[2]

fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(stage_vol_df['stage_m'], stage_vol_df['total_vol_cu_m'], 'bo', label='Observation')
plt.plot(dry_water_balance['stage(m)'], dry_water_balance['volume (cu.m)'], 'ro', label='Prediction')
plt.legend(loc='upper left')
plt.xlabel(r'\textbf{Stage} (m)')
plt.ylabel(r'\textbf{Volume} ($m^3$)')
# plt.xlim(-0.25, 2)
# plt.ylim(-100, 2000)
plt.title(r'Stage - Volume Relationship for 591 Check Dam')
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/stage_vol_linear_591')
# plt.show()
"""
Stage vs area linear relationship
"""
stage_area_df = pd.read_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/cont_area.csv',
                            sep=',', header=0, names=['sno', 'stage_m', 'total_area_sq_m'])
stage_area_df.drop('sno', inplace=True, axis=1)
#create empty column
dry_water_balance['ws_area(sq.m)'] = 0
# set stage as index
stage_area_df.set_index(stage_area_df['stage_m'], inplace=True)


for index, row in dry_water_balance.iterrows():
    # print 'index = %s' % index
    # print 'row = %s' % row
    obs_stage = row['stage(m)']  # observed stage
    # print obs_stage
    x1, x2 = find_range(stage_area_df['stage_m'].tolist(), obs_stage)
    x_diff = x2-x1
    y1 = stage_area_df['total_area_sq_m'][x1]
    y2 = stage_area_df['total_area_sq_m'][x2]
    y_diff = y2 - y1
    slope = y_diff/x_diff
    y_intercept = y2 - (slope*x2)
    # print x1, x2, stage_vol_df['total_vol_cu_m'][x1], stage_vol_df['total_vol_cu_m'][x2], slope
    dry_water_balance['ws_area(sq.m)'][index.strftime('%Y-%m-%d')] = (slope*obs_stage) + y_intercept

fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(stage_area_df['stage_m'], stage_area_df['total_area_sq_m'], 'bo')
plt.plot(dry_water_balance['stage(m)'], dry_water_balance['ws_area(sq.m)'], 'ro', label='Predicted water spread area')
plt.legend(loc='upper left')
plt.xlabel(r'\textbf{Stage} (m)')
plt.ylabel(r'\textbf{Area} ($m^2$)')
# plt.xlim(-0.25, 2)
# plt.ylim(-100, 2000)
plt.title(r'Stage - Surface Area Relationship for 591 Check Dam')
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/stage_area_linear_591')
# plt.show()


dry_water_balance['Evaporation (cu.m)'] = (dry_water_balance['Evaporation (mm/day)'] * 0.001) * dry_water_balance['ws_area(sq.m)']



# print dry_water_balance
cons_days = []
for d1, d2 in pairwise(dry_water_balance.index):
    # print d1, d2
    diff = abs((d2 - d1).days)
    # print diff
    if diff == 1:
        cons_days.append(d1)
        cons_days.append(d2)
# http://www.peterbe.com/plog/uniqifiers-benchmark

# function for taking unique values from list
def f2(seq):
    checked = []
    for e in seq:
        if e not in checked:
            checked.append(e)
    return checked

unique_dates = f2(cons_days)

# Select only consecutive dates for analysis
dry_wb_cons_day = dry_water_balance.ix[unique_dates]

# assume 0 initially
dry_wb_cons_day['change_storage(cu.m)'] = 0.000

for d1, d2 in pairwise(dry_wb_cons_day.index):
    # print d2.strftime('%Y-%m-%d')
    # print d1, d2
    diff = abs((d2 - d1).days)
    # print diff
    if diff == 1:
        dry_wb_cons_day['change_storage(cu.m)'][d2.strftime('%Y-%m-%d')] = dry_wb_cons_day['volume (cu.m)'][d2.strftime('%Y-%m-%d')] - dry_wb_cons_day['volume (cu.m)'][d1.strftime('%Y-%m-%d')]



dry_wb_cons_day['infiltration(cu.m)'] = -1*(dry_wb_cons_day['change_storage(cu.m)'] + dry_wb_cons_day['Evaporation (cu.m)'])

# removing stage values less than 5 cm or .05 metre
dry_wb_cons_day = dry_wb_cons_day[dry_wb_cons_day['stage(m)'] > 0.14]
# also remove zero change in storage values as these dont have previous date values
dry_wb_cons_day = dry_wb_cons_day[~(dry_wb_cons_day['change_storage(cu.m)'] == 0)]
"""
Correlation between stage and infiltration
"""
stage_cal = dry_wb_cons_day['stage(m)']     # x
inf_cal = dry_wb_cons_day['infiltration(cu.m)']       # y

# stage_inf_cal = polyfit(stage_cal, inf_cal, 2)    #
# po_stage_vol = np.polyfit(z_cal, area_cal, 2)
# f_stage_area = np.poly1d(po_stage_area)
# print np.poly1d(f_stage_area)
# print stage_area_cal
# print a_stage['polynomial'][0]
# coeff_stage_inf_cal = stage_inf_cal['polynomial']

#calculate new coefficientss

dry_wb_cons_day.to_csv('/media/kiruba/New Volume/milli_watershed/check_dam_evap/591/dry_wb_591.csv')
# plt.show()
# print dry_wb_cons_day
## plot of daily infiltration
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(dry_wb_cons_day.index, inf_cal, 'bo', label=r'Dry Infiltration ($m^3$)')
# plt.plot_date(rain_weather.index, np.random.normal(size=len(rain_weather.index)), 'ro', label='Rainy days')

# plt.ylabel(r'\textbf{Wind Speed}($Km/h$)')
plt.title(r"Infiltration for only Dry Days 591", fontsize=16)
plt.legend(loc='upper left')
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/dry_infiltration_591')
fig.autofmt_xdate()
# print dry_wb_cons_day
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(dry_wb_cons_day.index, dry_wb_cons_day['change_storage(cu.m)'], 'bo', label=r'Change in Storage ($m^3$)')
plt.legend(loc='upper left')
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/dry_change_storage_591')
fig.autofmt_xdate()
## dry day evaporation
# print dry_wb_cons_day
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(dry_wb_cons_day.index, dry_wb_cons_day['Evaporation (cu.m)'], 'bo', label=r'Evaporation ($m^3$)')
plt.legend(loc='upper left')
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/dry_evaporation_591')
fig.autofmt_xdate()
# plt.show()
##stacked bar plot
fig = plt.figure(figsize=(11.69, 8.27))
plt.bar(dry_wb_cons_day.index, dry_wb_cons_day['Evaporation (cu.m)'], 0.35, color='g', label=r'Evaporation ($m^3$)')
plt.bar(dry_wb_cons_day.index, inf_cal, 0.35, color='b', bottom=dry_wb_cons_day['Evaporation (cu.m)'], label=r'Infiltration ($m^3$)')
plt.legend(loc='upper left')
plt.title(r'Evaporation and Infiltration for Check dam 591')
plt.ylabel(r'$m^3$')
fig.autofmt_xdate()
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/dry_evaporation_inf_591')


##exponent fitting for stage vs infiltration
# http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.curve_fit.html

def func(h, alpha, beta):
    return alpha*(h**beta)

popt, pcov = curve_fit(func, stage_cal, inf_cal)

stage_cal_new = np.linspace(min(stage_cal), max(stage_cal), 50)
inf_cal_new = func(stage_cal_new, *popt)
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(stage_cal, inf_cal, 'bo', label=r'Observation')
plt.plot(stage_cal_new, inf_cal_new, 'r-', label='Prediction')
# plt.plot(stage_cal_new, inf_cal_new, 'b-', label=r'2\textsuperscript{nd} Degree Polynomial')
plt.legend(loc='upper left')
# plt.xlim([-0.2, 2.1])
# plt.ylim([-20, 2000])
plt.xlabel(r'\textbf{Stage} (m)')
plt.ylabel(r'\textbf{Infiltration} ($m^3/day$)')
# coeff_1 = format(66.66666666, '.2f')
# print coeff_1
# print type(coeff_stage_vol_cal[0])
plt.title(r'Stage - Infiltration Relationship for 591 Check Dam')
plt.text(x=0.15, y=11, fontsize=15, s=r'$Infiltration = {0:.2f}h^{{{1:.2f}}}$'.format(popt[0], popt[1]))
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/stage_inf_poly_2_deg_591')

# print popt
# print pcov
# print np.sqrt(np.diag(pcov))

"""
wet day water balance
"""
# merge dry days and wet day
dry_weather['status'] = 'D'
rain_weather['status'] = 'R'
#concatanete dry and wet day
water_balance_df = pd.concat([dry_weather, rain_weather], axis=0)
#sort based on date
water_balance_df.sort(inplace=True)
# select only rain, evporation and stage values
water_balance_df = water_balance_df[['Rain Collection (mm)', 'Evaporation (mm/day)', 'stage(m)', 'status']]
# create empty volume column
water_balance_df['volume (cu.m)'] = 0.000
# print(water_balance_df)

# removing stage values less than 14 cm or .14 metre
water_balance_df = water_balance_df[water_balance_df['stage(m)'] > 0.14]
# leave overflowing days
water_balance_df = water_balance_df[water_balance_df['stage(m)'] < 1.9]
# print stage_vol_df
# convert stage to volume

for index, row in water_balance_df.iterrows():
    obs_stage = row['stage(m)']
    print obs_stage
    x1, x2 = find_range(stage_vol_df['stage_m'].tolist(), obs_stage)
    x_diff = x2 - x1
    y1 = stage_vol_df['total_vol_cu_m'][x1]
    y2 = stage_vol_df['total_vol_cu_m'][x2]
    y_diff = y2 - y1
    slope = y_diff/x_diff
    y_intercept = y2 - (slope*x2)
    water_balance_df['volume (cu.m)'][index.strftime("%Y-%m-%d")] = (slope*obs_stage) + y_intercept


fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(stage_vol_df['stage_m'], stage_vol_df['total_vol_cu_m'], 'bo', label='Observation')
plt.plot(water_balance_df['stage(m)'], water_balance_df['volume (cu.m)'], 'ro', label='Prediction')
plt.legend(loc='upper left')
plt.xlabel(r'\textbf{Stage} (m)')
plt.ylabel(r'\textbf{Volume} ($m^3$)')
# plt.xlim(-0.25, 2)
# plt.ylim(-100, 2000)
plt.title(r'Stage - Volume Relationship for 591 Check Dam')
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/stage_vol_linear_comp_591')

"""
stage area relationship
"""
stage_area_df = pd.read_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/cont_area.csv',
                            sep=',', header=0, names=['sno', 'stage_m', 'total_area_sq_m'])
stage_area_df.drop('sno', inplace=True, axis=1)
#create empty column
water_balance_df['ws_area(sq.m)'] = 0.000
# set stage as index
stage_area_df.set_index(stage_area_df['stage_m'], inplace=True)

for index, row in water_balance_df.iterrows():
    # print 'index = %s' % index
    # print 'row = %s' % row
    obs_stage = row['stage(m)']  # observed stage
    # print obs_stage
    x1, x2 = find_range(stage_area_df['stage_m'].tolist(), obs_stage)
    x_diff = x2-x1
    y1 = stage_area_df['total_area_sq_m'][x1]
    y2 = stage_area_df['total_area_sq_m'][x2]
    y_diff = y2 - y1
    slope = y_diff/x_diff
    y_intercept = y2 - (slope*x2)
    # print x1, x2, stage_vol_df['total_vol_cu_m'][x1], stage_vol_df['total_vol_cu_m'][x2], slope
    water_balance_df['ws_area(sq.m)'][index.strftime('%Y-%m-%d')] = (slope*obs_stage) + y_intercept

fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(stage_area_df['stage_m'], stage_area_df['total_area_sq_m'], 'bo')
plt.plot(water_balance_df['stage(m)'], water_balance_df['ws_area(sq.m)'], 'ro', label='Predicted water spread area')
plt.legend(loc='upper left')
plt.xlabel(r'\textbf{Stage} (m)')
plt.ylabel(r'\textbf{Area} ($m^2$)')
# plt.xlim(-0.25, 2)
# plt.ylim(-100, 2000)
plt.title(r'Stage - Surface Area Relationship for 591 Check Dam')
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/stage_area_linear_comp_591')

water_balance_df['Evaporation (cu.m)'] = (water_balance_df['Evaporation (mm/day)']*.001) * water_balance_df['ws_area(sq.m)']
"""
change in storage
"""
#assume 0 initially
water_balance_df['change_storage(cu.m)'] = 0.000

#change in storage is today minus yesterday volume
for d1, d2 in pairwise(water_balance_df.index):
    water_balance_df['change_storage(cu.m)'][d2.strftime('%Y-%m-%d')] = water_balance_df['volume (cu.m)'][d2.strftime('%Y-%m-%d')]-water_balance_df['volume (cu.m)'][d1.strftime('%Y-%m-%d')]

# change in storage plot
colors = {'D': 'r', 'R': 'b'}
fig, ax1 = plt.subplots(figsize=(11.69, 8.27))
for t in ('D', 'R'):
    ax1.scatter(water_balance_df[water_balance_df['status'] == t].index, water_balance_df[water_balance_df['status'] == t]['change_storage(cu.m)'],
               color=colors[t],
               label=t,
               s=15,
               alpha=0.7)
# plt.plot_date(water_balance_df.index, water_balance_df['change_storage(cu.m)'], 'bo', label=r'Change in Storage ($m^3$)')
plt.legend(loc='best')
ax1.set_ylabel('Change in storage (cu.m)')
ax2 = ax1.twinx()
ax2.plot(water_level.index, water_level['stage(m)'], 'g-', label='Stage (m)')
ax2.set_ylabel('Stage (m)', color='g')
for t1 in ax2.get_yticklabels():
    t1.set_color('g')
plt.legend(loc='best')
fig.autofmt_xdate(rotation=90)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/whole_change_storage_591')
#take out dry days
water_balance_df_dry = water_balance_df[water_balance_df['status'] == 'D']
# calculate infiltration
water_balance_df_dry['infiltration(cu.m)'] = -1*(water_balance_df_dry['change_storage(cu.m)'] + water_balance_df_dry['Evaporation (cu.m)'])
# print water_balance_df_dry
# removing stage values less than 14 cm or .14 metre
water_balance_df_dry = water_balance_df_dry[water_balance_df_dry['stage(m)'] > 0.14]
water_balance_df_dry = water_balance_df_dry[water_balance_df_dry['infiltration(cu.m)'] > 0.0]
stage_cal = water_balance_df_dry['stage(m)']
inf_cal = water_balance_df_dry['infiltration(cu.m)']
# print water_balance_df_dry

def func(h, alpha, beta):
    return alpha*(h**beta)

popt, pcov = curve_fit(func, stage_cal, inf_cal)

print popt
print pcov
print np.sqrt(np.diag(pcov))

stage_cal_new = np.linspace(min(stage_cal), max(stage_cal), 50)
inf_cal_new = func(stage_cal_new, *popt)
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(stage_cal, inf_cal, 'bo', label=r'Observation')
plt.plot(stage_cal_new, inf_cal_new, 'r-', label='Prediction')
# plt.plot(stage_cal_new, inf_cal_new, 'b-', label=r'2\textsuperscript{nd} Degree Polynomial')
plt.legend(loc='upper left')
# plt.xlim([-0.2, 2.1])
# plt.ylim([-20, 2000])
plt.xlabel(r'\textbf{Stage} (m)')
plt.ylabel(r'\textbf{Infiltration} ($m^3/day$)')
# coeff_1 = format(66.66666666, '.2f')
# print coeff_1
# print type(coeff_stage_vol_cal[0])
plt.title(r'Stage - Infiltration Relationship for 591 Check Dam')
plt.text(x=0.15, y=11, fontsize=15, s=r'$Infiltration = {0:.2f}h^{{{1:.2f}}}$'.format(popt[0], popt[1]))
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/stage_inf_exp_dry_591')

