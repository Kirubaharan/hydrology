__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from spread import  spread
from scipy.optimize import curve_fit
import math
from matplotlib import rc
import time
import datetime
from datetime import timedelta
import scipy as sp
import meteolib as met
import evaplib
from bisect import bisect_left
import matplotlib as mpl
import Pysolar as ps
from mpl_toolkits.axes_grid1 import AxesGrid

#latex parameters
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
# print weather_df.columns.values[0]
# weather_df.columns.values[0] = 'Date_Time'
# print weather_df.head()
weather_df['Date_Time'] = pd.to_datetime(weather_df['Date_Time'], format=date_format)
weather_df.set_index(weather_df['Date_Time'], inplace=True)
# sort based on index
weather_df.sort_index(inplace=True)
# drop date time column
weather_df = weather_df.drop('Date_Time', 1)
# print weather_df.head()
# print weather_df['2014-b06-30']


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
# y_cal = [0.1, 0.4, 1, 1.6, 2.3, 2.8, 3 ]
x_cal = [2036, 2458, 3025, 4078, 5156, 5874, 6198]
a_stage = polyfit(x_cal, y_cal, 1)
# coefficients of polynomial are stored in following list
coeff_cal = a_stage['polynomial']
print coeff_cal


def read_correct_ch_dam_data(csv_file):
    """
    Function to read, calibrate and convert time format (day1 24:00:00
    to day 2 00:00:00) in check dam data
    :param csv_file:
    :return: calibrated and time corrected data
    """
    water_level = pd.read_csv(csv_file, skiprows=9, sep=',', header=0, names=['scan no', 'date', 'time', 'raw value', 'calibrated value'])
    water_level['calibrated value'] = (water_level['raw value'] *coeff_cal[0]) + coeff_cal[1]  # in cm
    water_level['calibrated value'] /= 100  # convert to metre
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
# print min(water_level_1.index), max(water_level_1.index)
# print water_level_1['stage(m)'][max(water_level_1.index)]
# print water_level_1.tail(20)
# water_level_1['stage(m)'][max(water_level_1.index)] = 0.5*(water_level_1['stage(m)'][max(water_level_1.index)])
block_2 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525_008_002.CSV'
water_level_2 = read_correct_ch_dam_data(block_2)
# print water_level_2.head()
# print water_level_2.tail()
block_3 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525_008_003.CSV'
water_level_3 = read_correct_ch_dam_data(block_3)
# print water_level_3.head()
# print water_level_3.tail()
block_4 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525_008_004.CSV'
water_level_4 = read_correct_ch_dam_data(block_4)
# print water_level_4.head()
# print water_level_4.tail()
water_level = pd.concat([water_level_1, water_level_2, water_level_3, water_level_4], axis=0)
# print water_level.head(20)
water_level = water_level['2014-05-14 18:30:00':'2014-09-10 23:30:00']
# water_level = np.around(water_level, 2)
print water_level.head()
"""
Fill in missing values interpolate
"""
new_index = pd.date_range(start='2014-05-14 18:30:00', end='2014-09-10 23:30:00', freq='30min' )
water_level = water_level.reindex(new_index, method=None)
water_level = water_level.interpolate(method='time')
water_level.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/stage_30min.CSV')
"""
Join weather and rain data
"""
weather_df = weather_df.join(rain_df, how='right')
weather_df = weather_df['2014-05-14':'2014-09-10']
# print weather_df['2014-06-30']
# weather_df = weather_df[min(water_level.index): max(water_level.index)]
weather_df = weather_df.join(water_level, how='right')
# print weather_df['2014-06-30']
# print weather_df.head(20)

"""
Evaporation from open water
Equation according to J.D. Valiantzas (2006). Simplified versions
for the Penman evaporation equation using routine weather data.
J. Hydrology 331: 690-702. Following Penman (1948,1956). Albedo set
at 0.06 for open water.
Input (measured at 2 m height):
        - airtemp: (array of) average air temperatures [Celsius]
        - rh: (array of)  average relative humidity [%]
        - airpress: (array of) average air pressure data [Pa]
        - Rs: (array of) incoming solar radiation [J/m2/day]
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
weather_df['AirPr(Pa)'] = air_p_pa
"""
Half hourly Extraterrestrial Radiation Calculation(J/m2/30min)
"""

SC_default = 1367.0 # Solar constant in W/m^2 is 1367.0.


def extraterrestrial_irrad(local_datetime, latitude_deg, longitude_deg):
    """
    Calculates extraterrestrial radiation in MJ/m2/timeperiod
    :param local_datetime: datetime object
    :param latitude_deg: in decimal degree
    :param longitude_deg: in decimal degree
    :return: Extra terrestrial radiation in MJ/m2/timeperiod
    """

    S = 0.0820  # MJ m-2 min-1
    lat_rad = latitude_deg*(math.pi/180)
    day = ps.solar.GetDayOfYear(local_datetime)
    hour = float(local_datetime.hour)
    minute = float(local_datetime.minute)
    b = ((2*math.pi)*(day-81))/364
    sc = 0.1645*(math.sin(2*b)) - 0.1255*(math.cos(b)) - 0.025*(math.sin(b))  # seasonal correction in hour
    lz = 270   # for India longitude of local time zone in degrees west of greenwich
    lm = (180+(180-longitude_deg))  # longitude of measurement site
    t = (hour + (minute/60)) - 0.25
    t1 = 0.5  # 0.5 for 30 minute 1 for hourly period
    w = (math.pi/12)*((t + (0.0667*(lz-lm))+ sc) - 12)
    w1 = w - ((math.pi*t1)/24)  # solar time angle at beginning of period [rad]
    w2 = w + ((math.pi*t1)/24)  # solar time angle at end of period [rad]
    dr = 1 + (0.033*math.cos((2*math.pi*day)/365))  # inverse relative distance Earth-Sun
    dt = 0.409*math.sin(((2*math.pi*day)/365) - 1.39) # solar declination in radian
    ws = math.acos(-math.tan(lat_rad)*math.tan(dt))
    if (w > ws) or (w < -ws):
        Rext = 0.0
    else:
        Rext = ((12*60)/math.pi)*S*dr*(((w2-w1)*math.sin(lat_rad)*math.sin(dt))+(math.cos(lat_rad)*math.cos(dt)*(math.sin(w2) - math.sin(w1))))  # MJm-2(30min)-1
    return Rext

ch_591_lat = 13.260196
ch_591_long = 77.512085
weather_df['Rext (MJ/m2/30min)'] = 0.000
for i in weather_df.index:
    weather_df['Rext (MJ/m2/30min)'][i.strftime('%Y-%m-%d %H:%M:%S')] = (extraterrestrial_irrad(local_datetime=i,
                                                                                                latitude_deg=ch_591_lat,
                                                                                                longitude_deg=ch_591_long))


# weather_df['Rext (MJ/m2/30min)'] =
# weather_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/weather.csv')
"""
wind speed from km/h to m/s
1 kmph = 0.277778 m/s
"""
weather_df['Wind Speed (mps)'] = weather_df['Wind Speed (kmph)'] * 0.277778
"""
Radiation unit conversion
"""
# the radiation units are actually in W/m2 and
#  not in W/mm2 as given by weather station,
# so multiply with 30*60 seconds
# to convert to MJ divide by 10^6
weather_df['Solar Radiation (MJ/m2/30min)'] = (weather_df['Solar Radiation (W/mm2)'] * 1800)/(10**6)
"""
Average Temperature Calculation
"""
weather_df['Average Temp (C)'] = 0.5*(weather_df['Min Air Temperature (C)'] + weather_df['Max Air Temperature (C)'])

weather_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/weather.csv')


"""
Open water evaporation function for half hour
Modified from evaplib.py
http://python.hydrology-amsterdam.nl/moduledoc/index.html#module-evaplib
"""


def delta_calc(airtemp):
    """
    Calculates slope of saturation vapour pressure curve at air temperature [kPa/Celsius]
    http://www.fao.org/docrep/x0490e/x0490e07.htm
    :param airtemp: Temperature in Celsius
    :return: slope of saturation vapour pressure curve [kPa/Celsius]
    """
    l = sp.size(airtemp)
    if l < 2:
        temp_kelvin = airtemp + 237.3
        b = 0.6108*(math.exp((17.27*airtemp)/temp_kelvin))
        delta = (4098*b)/(temp_kelvin**2)
    else:
        delta = sp.zeros(l)
        for i in range(0, l):
            temp_kelvin = airtemp[i] + 237.3
            b = 0.6108*(math.exp(17.27*airtemp[i])/temp_kelvin)
            delta[i] = (4098*b)/(temp_kelvin**2)
    return delta


def half_hour_E0(airtemp = sp.array([]),
                 rh = sp.array([]),
                 airpress = sp.array([]),
                 Rs = sp.array([]),
                 Rext = sp.array([]),
                 u =sp.array([]),
                 Z=0.0):
    """
    Function to calculate daily Penman open water evaporation (in mm/30min).
    Equation according to
    Shuttleworth, W. J. 2007. "Putting the 'Vap' into Evaporation."
    Hydrology and Earth System Sciences 11 (1): 210-44. doi:10.5194/hess-11-210-2007.

    :param airtemp: average air temperature [Celsius]
    :param rh: relative humidity[%]
    :param airpress: average air pressure[Pa]
    :param Rs: Incoming solar radiation [MJ/m2/30min]
    :param Rext: Extraterrestrial radiation [MJ/m2/30min]
    :param u: average wind speed at 2 m from ground [m/s]
    :param Z: site elevation, default is zero [metre]
    :return: Penman open water evaporation values [mm/30min]
    """
     # Set constants
    albedo = 0.06  # open water albedo
    # Stefan boltzmann constant = 5.670373*10-8 J/m2/k4/s
    # http://en.wikipedia.org/wiki/Stefan-Boltzmann_constant
    # sigma = 5.670373*(10**-8)  # J/m2/K4/s
    sigma = (1.02066714*(10**-10))  #Stefan Boltzmann constant MJ/m2/K4/30min
    # Calculate Delta, gamma and lambda
    DELTA = delta_calc(airtemp)     # [Kpa/C]
    # Lambda = met.L_calc(airtemp)/(10**6) # [MJ/Kg]
    # gamma = met.gamma_calc(airtemp, rh, airpress)/1000
    # Lambda = 2.501 -(0.002361*airtemp)     # [MJ/kg]
    # gamma = (0.0016286 *(airpress/1000))/Lambda
    # Calculate saturated and actual water vapour pressure
    es = met.es_calc(airtemp)  # [Pa]
    ea = met.ea_calc(airtemp,rh)  # [Pa]
    #Determine length of array
    l = sp.size(airtemp)
    #Check if we have a single value or an array
    if l < 2:
        Lambda = 2.501 -(0.002361*airtemp)     # [MJ/kg]
        gamma = (0.0016286 *(airpress/1000))/Lambda
        Rns = (1.0 - albedo)* Rs  # shortwave component [MJ/m2/30min]
        #calculate clear sky radiation Rs0
        Rs0 = (0.75+(2E-5*Z))*Rext
        f = (1.35*(Rs/Rs0))-0.35
        epsilom = 0.34-(-0.14*sp.sqrt(ea/1000))
        Rnl = f*epsilom*sigma*(airtemp+273.16)**4  # Longwave component [MJ/m2/30min]
        Rnet = Rns - Rnl
        Ea = (1 + (0.536*u))*((es/1000)-(ea/1000))
        E0 = ((DELTA*Rnet) + gamma*(6.43*Ea))/(Lambda*(DELTA+gamma))
    else:
        # Inititate output array
        E0 = sp.zeros(l)
        Rns = sp.zeros(l)
        Rs0 = sp.zeros(l)
        f = sp.zeros(l)
        epsilom = sp.zeros(l)
        Rnl = sp.zeros(l)
        Rnet = sp.zeros(l)
        Ea = sp.zeros(l)
        Lambda = sp.zeros(l)
        gamma = sp.zeros(l)
        for i in range(0,l):
            Lambda[i] = 2.501 -(0.002361*airtemp[i])
            gamma[i] = (0.0016286 *(airpress[i]/1000))/Lambda[i]
            # calculate longwave radiation (MJ/m2/30min)
            Rns[i] = (1.0 - albedo) * Rs[i]
            # calculate clear sky radiation Rs0
            Rs0[i] = (0.75 + (2E-5*Z))
            f[i] = (1.35*(Rs[i]/Rs0[i]))-0.35
            epsilom[i] = 0.34-(-0.14*sp.sqrt(ea[i]/1000))
            Rnl[i] = f[i]*epsilom[i]*sigma*(airtemp[i]+273.16)**4  # Longwave component [MJ/m2/30min]
            Rnet[i] = Rns[i] - Rnl[i]
            Ea[i] = (1 + (0.536*u[i]))*((es[i]/1000)-(ea[i]/1000))
            E0[i] = ((DELTA[i]*Rnet[i]) + gamma[i]*(6.43*Ea[i]))/(Lambda[i]*(DELTA[i]+gamma[i]))
    return E0


"""
Half hourly Evaporation calculation
"""
airtemp = weather_df['Average Temp (C)']
hum = weather_df['Humidity (%)']
airpress = weather_df['AirPr(Pa)']
rs = weather_df['Solar Radiation (MJ/m2/30min)']
rext = weather_df['Rext (MJ/m2/30min)']
wind_speed = weather_df['Wind Speed (mps)']
weather_df['Evaporation (mm/30min)'] = half_hour_E0(airtemp=airtemp, rh=hum, airpress=airpress,
                                                    Rs=rs, Rext=rext, u=wind_speed, Z=z)
"""
Plot Evaporation
"""
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_df.index, weather_df['Evaporation (mm/30min)'], '-g', label='Evaporation (mm/30min)')
plt.ylabel(r'\textbf{Evaporation ($mm/30min$)}')
fig.autofmt_xdate(rotation=90)
plt.title(r"Daily Evaporation for Check Dam - 591", fontsize=20)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_591/evaporation_591_30min')
# bar plots
weather_sel_df = weather_df['2014-05-20':'2014-05-22']
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_sel_df.index, weather_sel_df['Evaporation (mm/30min)'], '-g')
fig.autofmt_xdate(rotation=90)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_591/evaporation_591_may20_22')
# plt.show()
"""
Remove Duplicates
"""
# check for duplicates
# df2 = dry_weather.groupby(level=0).filter(lambda x: len(x) > 1)
# print(df2)
weather_df['index'] = weather_df.index
weather_df.drop_duplicates(subset='index', take_last=True, inplace=True)
del weather_df['index']
weather_df = weather_df.sort()
"""
Stage Volume relation estimation from survey data
"""
# neccessary functions


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2,s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

"""
Select data where stage is available
"""
weather_stage_avl_df = weather_df[min(water_level.index):max(water_level.index)]

"""
Convert observed stage to volume by linear interpolation
"""
# set stage as index
stage_vol_df = pd.read_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/stage_vol.csv',
                            sep=',', header=0, names=['sno', 'stage_m', 'total_vol_cu_m'])
# print stage_vol_df

stage_vol_df.drop('sno', inplace=True, axis=1)
stage_vol_df.set_index(stage_vol_df['stage_m'], inplace=True)
# function to find containing intervals


def find_range(array, ab):
    if ab < max(array):
        start = bisect_left(array, ab)
        return array[start-1], array[start]
    else:
        return min(array), max(array)


# print weather_stage_avl_df.head()
water_balance_df = weather_stage_avl_df[['Rain Collection (mm)', 'Evaporation (mm/30min)', 'stage(m)']]
# print find_range(stage_vol_df['stage_m'].tolist(), max(water_balance_df['stage(m)']))
water_balance_df['volume (cu.m)'] = 0.000
stage_cutoff = 0.1      # metre

for index, row in water_balance_df.iterrows():
    # print index
    obs_stage = row['stage(m)']  # observed stage
    if obs_stage > stage_cutoff:
        x1, x2 = find_range(stage_vol_df['stage_m'].tolist(), obs_stage)
        x_diff = x2-x1
        y1 = stage_vol_df['total_vol_cu_m'][x1]
        y2 = stage_vol_df['total_vol_cu_m'][x2]
        y_diff = y2 - y1
        slope = y_diff/x_diff
        y_intercept = y2 - (slope*x2)
        water_balance_df['volume (cu.m)'][index.strftime('%Y-%m-%d %H:%M:%S')] = (slope*obs_stage) + y_intercept


"""
Overflow
"""
full_vol = stage_vol_df['total_vol_cu_m'][1.9]
# print full_vol
water_balance_df['overflow(cu.m)'] = 0.000
for index, row in water_balance_df.iterrows():
    obs_vol = row['volume (cu.m)']
    if obs_vol > full_vol:
        # print obs_vol
        water_balance_df['overflow(cu.m)'][index.strftime('%Y-%m-%d %H:%M:%S')] = obs_vol - full_vol
        water_balance_df['volume (cu.m)'][index.strftime('%Y-%m-%d %H:%M:%S')] = full_vol
        water_balance_df['stage(m)'][index.strftime('%Y-%m-%d %H:%M:%S')] = 1.9

# start from May 15
# water_balance_df = water_balance_df["2014-05-15":]
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
    if obs_stage > stage_cutoff:
        x1, x2 = find_range(stage_area_df['stage_m'].tolist(), obs_stage)
        x_diff = x2-x1
        y1 = stage_area_df['total_area_sq_m'][x1]
        y2 = stage_area_df['total_area_sq_m'][x2]
        y_diff = y2 - y1
        slope = y_diff/x_diff
        y_intercept = y2 - (slope*x2)
        water_balance_df['ws_area(sq.m)'][index.strftime('%Y-%m-%d %H:%M:%S')] = (slope*obs_stage) + y_intercept

# fig = plt.figure()
# plt.plot(stage_area_df['stage_m'], stage_area_df['total_area_sq_m'], 'bo')
# # stage_cal_new = np.linspace(min(stage_area_df['stage_m']), max(stage_area_df['stage_m']), 50)
# # area_cal_new =
# # plt.plot(water_balance_df['stage(m)'], water_balance_df['ws_area(sq.m)'], 'ro-')
# plt.show()
"""
Evaporation Volume estimation
"""
water_balance_df['Evaporation (cu.m)'] = 0.0
for index in water_balance_df.index:
    water_balance_df['Evaporation (cu.m)'][index.strftime(date_format)] = (water_balance_df['Evaporation (mm/30min)'][index.strftime(date_format)] * 0.001) * water_balance_df['ws_area(sq.m)'][index.strftime(date_format)]

water_balance_df['change_storage(cu.m)'] = 0.000
time_interval = 1800 # 1800 seconds =  30 min
for d1, d2 in pairwise(water_balance_df.index):
    diff = (d2-d1).seconds
    if diff == time_interval:
        d1_storage = water_balance_df['volume (cu.m)'][d1.strftime(date_format)]
        d2_storage = water_balance_df['volume (cu.m)'][d2.strftime(date_format)]
        water_balance_df['change_storage(cu.m)'][d2.strftime(date_format)] = d2_storage - d1_storage

"""
Separate out no inflow time period
change in storage = Inflow - Outflow
Change in storage will be negative when Inflow = 0 and when Inflow is less than Outflow
"""
# fig, ax1 = plt.subplots(figsize=(11.69, 8.27))
# line1 = ax1.plot(water_balance_df.index, water_balance_df['change_storage(cu.m)'], 'r-')
# ax1.hlines(0, min(water_balance_df.index), max(water_balance_df.index), 'g', linewidth=2)
# ax1.tick_params(axis='x',
#                 which='both',
#                 top='off',
#                 bottom='off',
#                 labeltop='off')
# ax2 = ax1.twinx()
# line2 = ax2.plot(water_balance_df.index, water_balance_df['stage(m)'], 'b-')
# fig.autofmt_xdate(rotation=90)
# plt.show()
water_balance_df['status'] = 'Y'
three_hour = 10800
no_rain_df = water_balance_df[water_balance_df['Rain Collection (mm)'] == 0]
for index in no_rain_df.index:
    initial_time_stamp = min(water_balance_df.index) + timedelta(seconds=three_hour)
    if index > initial_time_stamp:
        start_date = index - timedelta(seconds=three_hour)
        three_hour_rain_df = water_balance_df['Rain Collection (mm)'][start_date.strftime('%Y-%m-%d %H:%M:%S'):index.strftime('%Y-%m-%d %H:%M:%S')]
        sum_df = three_hour_rain_df.sum(axis=0)
        if (sum_df == 0) and (water_balance_df['change_storage(cu.m)'][index.strftime(date_format)] < 0):
            water_balance_df['status'][index.strftime('%Y-%m-%d %H:%M:%S')] = "N"

#
dry_water_balance_df = water_balance_df[water_balance_df['status'] == "N"]
rain_water_balance_df = water_balance_df[water_balance_df['status'] == "Y"]
# dry_water_balance_df = dry_water_balance_df[dry_water_balance_df['change_storage(cu.m)'] < 0]
# rain_water_balance_df = water_balance_df[water_balance_df['change_storage(cu.m)'] >= 0]
"""
# Calculate infiltration
# """
# # calculate infiltration
# dry_water_balance_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/dry_wb_check.CSV')
# # print dry_water_balance_df.head()
print 's'
dry_water_balance_df['infiltration(cu.m)'] = 0.000
delta_s = water_balance_df['change_storage(cu.m)']
evap = water_balance_df['Evaporation (cu.m)']
outflow = water_balance_df['overflow(cu.m)']

# for index, row in dry_water_balance_df.iterrows():
#     if index > min(dry_water_balance_df.index):
#         t_1 = index - timedelta(seconds=1800)
#         if t_1 < max(dry_water_balance_df.index):
#             dry_water_balance_df['infiltration(cu.m)'][index.strftime(date_format)] = -1.0*(delta_s[index.strftime(date_format)] + evap[t_1.strftime(date_format)] + outflow[t_1.strftime(date_format)])

for index in dry_water_balance_df.index:
    dry_water_balance_df['infiltration(cu.m)'][index.strftime(date_format)] = -1.0*(delta_s[index.strftime(date_format)] + evap[index.strftime(date_format)] + outflow[index.strftime(date_format)])

fig = plt.figure()
plt.plot(dry_water_balance_df.index, dry_water_balance_df['infiltration(cu.m)'], 'r-')
fig.autofmt_xdate(rotation=90)
plt.show()

dry_water_balance_df['infiltration rate (m/30min)'] = 0.0
for i in dry_water_balance_df.index:
   dry_water_balance_df['infiltration rate (m/30min)'][i.strftime("%Y-%m-%d %H:%M:%S")] = dry_water_balance_df['infiltration(cu.m)'][i.strftime("%Y-%m-%d %H:%M:%S")] / dry_water_balance_df['ws_area(sq.m)'][i.strftime("%Y-%m-%d %H:%M:%S")]

dry_water_balance_df = dry_water_balance_df[dry_water_balance_df['stage(m)'] > stage_cutoff]
# dry_water_balance_df = dry_water_balance_df[dry_water_balance_df['chanage_storage(cu.m)']]
# dry_water_balance_df = dry_water_balance_df[dry_water_balance_df['infiltration rate (m/30min)'] > 0]
dry_water_balance_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/wb_30min/dry_wb_30min.csv')

"""
Fitting exponential function
"""
stage_cal = dry_water_balance_df['stage(m)']
inf_cal = dry_water_balance_df['infiltration rate (m/30min)']
fig = plt.figure()
plt.plot(stage_cal, inf_cal, 'bo')
plt.show()


def func(h, alpha, beta):
    return (alpha*(h**beta))

popt, pcov = curve_fit(f=func, xdata=stage_cal, ydata=inf_cal)

stage_cal_new = np.linspace(min(stage_cal), max(stage_cal), 50)
inf_cal_new = func(stage_cal_new, *popt)
fig = plt.figure(figsize=(11.69, 8.27))
# cmap, norm = mpl.colors.from_levels_and_colors([0.1, 0.5, 1, 1.5, 2.0], ['red', 'yellow', 'green', 'blue'])
plt.plot(stage_cal, inf_cal, 'bo', label=r'Observation') #c=dry_water_balance_df['stage(m)'], cmap=cmap, norm=norm)
plt.plot(stage_cal_new, inf_cal_new, 'r-', label='Prediction')
plt.vlines(1.9, 0, max(inf_cal), 'g')
plt.hlines(0, min(stage_cal), max(stage_cal), 'y')
plt.legend(loc='upper right')
plt.xlabel(r'\textbf{Stage} (m))')
plt.ylabel(r'\textbf{Infiltration Rate} ($m/30min$)')
plt.title(r"No inflow day's stage - infiltration relationship for 591 check dam")
plt.text(x=0.3, y=.004, fontsize=15, s=r'$Infiltration = {0:.2f}{{h_{{av}}}}^{{{1:.2f}}}$'.format(popt[0], popt[1]))
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_591/half_hour_wb/dry_infil_591_30min')
plt.show()
"""
Inflow day's infiltration
"""
rain_water_balance_df['infiltration(cu.m)'] = 0.0
rain_water_balance_df['infiltration rate (m/30min)'] = 0.0
for i in rain_water_balance_df.index:
    rain_water_balance_df['infiltration rate (m/30min)'][i.strftime(date_format)] = (popt[0]*(rain_water_balance_df['stage(m)'][i.strftime(date_format)]**popt[1]))
for i in rain_water_balance_df.index:
    rain_water_balance_df['infiltration(cu.m)'][i.strftime(date_format)] = rain_water_balance_df['infiltration rate (m/30min)'][i.strftime(date_format)] * rain_water_balance_df['ws_area(sq.m)'][i.strftime(date_format)]

fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(rain_water_balance_df['stage(m)'], rain_water_balance_df['infiltration(cu.m)'], 'bo', label='Predicted Infiltration' )
plt.vlines(1.9, 0, 100, 'g')
# plt.xlim([-1, 2.0])
plt.legend(loc='upper left')
plt.xlabel(r'\textbf{Stage} (m)')
plt.ylabel(r'\textbf{Infiltration} ($m^3/day$)')
plt.title(r"Inflow day's stage - infiltration relationship for 591 check dam")
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_591/half_hour_wb/rain_inf_591_30min')
# plt.show()
"""
Inflow calculation
"""
rain_water_balance_df['Inflow (cu.m)'] = 0.000
dry_water_balance_df['Inflow (cu.m)'] = 0.000
delta_s_rain = rain_water_balance_df['change_storage(cu.m)']
inf_rain = rain_water_balance_df['infiltration(cu.m)']
evap_rain = rain_water_balance_df['Evaporation (cu.m)']
outflow_rain = rain_water_balance_df['overflow(cu.m)']
for i in rain_water_balance_df.index:
    rain_water_balance_df['Inflow (cu.m)'][i.strftime(date_format)] = (delta_s_rain[i.strftime(date_format)] + inf_rain[i.strftime(date_format)] + evap_rain[i.strftime(date_format)] + outflow_rain[i.strftime(date_format)])

fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(rain_water_balance_df['Rain Collection (mm)'], rain_water_balance_df['Inflow (cu.m)'], 'bo', label='Predicted Inflow' )
# # plt.vlines(1.9, 0, 100, 'g')
# # plt.xlim([-1, 2.0])
# # plt.legend(loc='upper left')
plt.xlabel(r'\textbf{Rainfall} (mm)')
plt.ylabel(r'\textbf{Inflow} ($m^3/30min$)')
plt.title(r"Inflow day's Rainfall-Inflow relationship for 591 check dam")
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_591/half_hour_wb/rain_inflow_591_30min')

"""
Merge dry and wet df
"""
merged_water_balance = pd.concat([dry_water_balance_df, rain_water_balance_df])
merged_water_balance.sort_index(inplace=True)
merged_water_balance = merged_water_balance[merged_water_balance['stage(m)'] > 0.1]
# merged_water_balance = merged_water_balance.resample('D', how=np.sum)
merged_water_balance.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/wb_30min/wb_30min.csv')
"""
Evaporation vs infiltration
"""
fig, ax1 = plt.subplots(figsize=(11.69, 8.27))
line1 = ax1.plot(merged_water_balance.index, merged_water_balance['Evaporation (cu.m)'], 'r-', label=r"\textbf{Evaporation ($m^3/day$)}")
# plt.title("Evaporation vs Infiltration for Check dam 591")
for t1 in ax1.get_yticklabels():
    t1.set_color('r')
ax2 = ax1.twiny()
line2 = ax2.plot(merged_water_balance.index, merged_water_balance['infiltration(cu.m)'],'g-', label=r"\textbf{Infiltration ($m^3/day$}")
for t1 in ax2.get_yticklabels():
    t1.set_color('g')
lns = [line1, line2]
lab = [r"\textbf{Evaporation ($m^3/30min$)}", r"\textbf{Infiltration ($m^3/30min$}" ]
# ax2.legend(lns, lab, loc='upper center', fancybox=True, ncol=2, bbox_to_anchor=(0.5, 1.15))
fig.autofmt_xdate(rotation=90)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_591/half_hour_wb/evap_infilt_591_30min')
plt.show()
