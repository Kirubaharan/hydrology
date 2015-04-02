__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from spread import  spread
from scipy.optimize import curve_fit
import math
from matplotlib import rc
import email.utils as eutils
import time
import datetime
from datetime import timedelta
import scipy as sp
import meteolib as met
import evaplib
from bisect import bisect_left
import matplotlib as mpl
import Pysolar as ps

# latex parameters
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)

"""
Variables
"""
full_stage = 0.66  # check dam height, above this it is assumed check dam will overflow
date_format = '%Y-%m-%d %H:%M:%S'
daily_format = '%Y-%m-%d'
resolution_ody = 0.0008
stage_cutoff = 0.1

# ------------------------------------------------------------------#
# Weather file
weather_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/corrected_weather.csv'
# Rain file
rain_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/corrected_rain.csv'

# convert to pandas dataframe
weather_df = pd.read_csv(weather_file, sep=',', header=0)
# set index

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
# print weather_df['2014-06-30']


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
y_cal = np.array([100, 400, 1000, 1600, 2250, 2750])
x_cal = np.array([2114, 2464, 3145, 3799, 4550, 5095])
a_stage = polyfit(x_cal, y_cal, 1)
# coefficients of polynomial are stored in following list
coeff_cal = a_stage['polynomial']


def myround(a, decimals=1):
     return np.around(a-10**(-(decimals+5)), decimals=decimals)


def read_correct_ch_dam_data(csv_file):
    """
    Function to read, calibrate and convert time format (day1 24:00:00
    to day 2 00:00:00) in check dam data
    :param csv_file:
    :return: calibrated and time corrected data
    """
    water_level = pd.read_csv(csv_file, skiprows=9, sep=',', header=0, names=['scan no', 'date', 'time', 'raw value', 'calibrated value'])
    water_level['calibrated value'] = (water_level['raw value'] *coeff_cal[0]) + coeff_cal[1] #in cm
    # water_level['calibrated value'] = np.round(water_level['calibrated value']/resolution_ody)*resolution_ody
    water_level['calibrated value'] /= 1000.0
    water_level['calibrated value'] = myround(a=water_level['calibrated value'], decimals=3)
    # #change the column name
    water_level.columns.values[4] = 'stage(m)'
    # print water_level.head()

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
    for index, row in water_level.iterrows():
        # print row
        obs_stage = row['stage(m)']
        if obs_stage < stage_cutoff:
            water_level['stage(m)'][index.strftime(date_format)] = 0.0

    water_level.drop(['scan no', 'date', 'time', 'date_time'], inplace=True, axis=1)

    return water_level


## Read check dam data
block_1 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/463/3605_023_001_03_12_2014.CSV'
water_level_1 = read_correct_ch_dam_data(block_1)
block_2 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/463/3605_023_003_23_12_2014.CSV'
water_level_2 = read_correct_ch_dam_data(block_2)
# block_3 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525_008_003.CSV'
# water_level_3 = read_correct_ch_dam_data(block_3)
# block_4 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525_008_004.CSV'
# water_level_4 = read_correct_ch_dam_data(block_4)
# block_5 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525_008_005.CSV'
# water_level_5 = read_correct_ch_dam_data(block_5)
# block_6 = '/media/kiruba/New Volume/ACCUWA_Data/check_dam_water_level/2525_008_006.CSV'
# water_level_6 = read_correct_ch_dam_data(block_6)
water_level = pd.concat([water_level_1, water_level_2], axis=0)
# water_level = pd.concat([water_level_1, water_level_2, water_level_3, water_level_4, water_level_5, water_level_6], axis=0)
water_level = water_level.sort()
# water_level = water_level['2014-05-14 18:30:00':'2014-09-10 23:30:00']
"""
Fill in missing values interpolate
"""
start_time = min(water_level.index)
end_time = max(water_level.index)
new_index = pd.date_range(start=start_time, end=end_time, freq='30min')
water_level = water_level.reindex(new_index, method=None)
water_level = water_level.interpolate(method='time')
new_index = pd.date_range(start=start_time.strftime('%Y-%m-%d %H:%M'), end=end_time.strftime('%Y-%m-%d %H:%M'), freq='30Min')
water_level = water_level.set_index(new_index)
water_level.index.name = 'Date'
# raise SystemExit(0)
# water_level_test = water_level["2014-06-15": "2014-06-22"]
fig = plt.figure()
plt.plot(x_cal, y_cal/1000.0, 'b-o')
plt.plot(water_level['raw value'], water_level['stage(m)'], 'ro')
# plt.plot(water_level_test.index, water_level_test['stage(m)'], 'b-o')
# for label, x, y in zip(water_level_test['overflow(cu.m)'], water_level_test.index, water_level_test['stage(m)']):
#     plt.annotate(label, xy = (x, y), xytext = (-20, 20),
#         textcoords = 'offset points', ha = 'right', va = 'bottom',
#         bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#         arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

# plt.hlines(1.9, xmin=min(water_level.index), xmax=max(water_level.index))
# plt.hlines(1.91, xmin=min(water_level.index), xmax=max(water_level.index))
plt.show()
# raise SystemExit(0)
"""
Join weather and rain data
"""
weather_df = weather_df.join(rain_df, how='right')
weather_df = weather_df[min(water_level.index).strftime(daily_format): max(water_level.index).strftime(daily_format)]
# print weather_df['2014-06-30']
# weather_df = weather_df[min(water_level.index): max(water_level.index)]
weather_df = weather_df.join(water_level, how='right')
# print weather_df['2014-06-30']
# print weather_df.head(20)
# weather_df.to_csv()
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

ch_463_lat = 13.360354
ch_463_long = 77.527267
weather_df['Rext (MJ/m2/30min)'] = 0.000
for i in weather_df.index:
    weather_df['Rext (MJ/m2/30min)'][i.strftime('%Y-%m-%d %H:%M:%S')] = (extraterrestrial_irrad(local_datetime=i,
                                                                                                latitude_deg=ch_463_lat,
                                                                                                longitude_deg=ch_463_long))


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
weather_df['Solar Radiation (MJ/m2/30min)'] = (weather_df['Solar Radiation (W/m2)'] * 1800)/(10**6)
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
        temp = airtemp + 237.3
        b = 0.6108*(math.exp((17.27*airtemp)/temp))
        delta = (4098*b)/(temp**2)
    else:
        delta = sp.zeros(l)
        for i in range(0, l):
            temp = airtemp[i] + 237.3
            b = 0.6108*(math.exp(17.27*airtemp[i])/temp)
            delta[i] = (4098*b)/(temp**2)
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
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_463/evaporation_463_30min')
# bar plots
# weather_sel_df = weather_df['2014-05-20':'2014-05-22']
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_df.index, weather_df['Evaporation (mm/30min)'], '-g')
fig.autofmt_xdate(rotation=90)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_463/new_evaporation_463_')
# plt.show()
# """
# Remove Duplicates
# """
# # check for duplicates
# # df2 = dry_weather.groupby(level=0).filter(lambda x: len(x) > 1)
# # print(df2)
# weather_df['index'] = weather_df.index
# weather_df.drop_duplicates(subset='index', take_last=True, inplace=True)
# del weather_df['index']
# weather_df = weather_df.sort()
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
stage_vol_df = pd.read_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/stage_vol.csv',
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
# stage_cutoff = 0.1
for index, row in water_balance_df.iterrows():
    # print index
    obs_stage = row['stage(m)']  # observed stage
    if obs_stage >= stage_cutoff:
        x1, x2 = find_range(stage_vol_df['stage_m'].tolist(), obs_stage)
        x_diff = x2-x1
        y1 = stage_vol_df['total_vol_cu_m'][x1]
        y2 = stage_vol_df['total_vol_cu_m'][x2]
        y_diff = y2 - y1
        slope = y_diff/x_diff
        y_intercept = y2 - (slope*x2)
        water_balance_df['volume (cu.m)'][index.strftime('%Y-%m-%d %H:%M:%S')] = (slope*obs_stage) + y_intercept

# fig = plt.figure(figsize=(11.69, 8.27))
# plt.plot_date(water_balance_df.index, water_balance_df['volume (cu.m)'], '-g')
# plt.hlines(stage_vol_df['total_vol_cu_m'][1.9], min(water_balance_df.index), max(water_balance_df.index))
# plt.title('before overflow correction')
water_balance_df['pumping status'] = 0.00
# water_balance_df['pumping (cu.m)'] = 0.00
"""
Pumping
"""
for index, row in water_balance_df.iterrows():
    if index > min(water_balance_df.index):
        previous_time = index - timedelta(seconds=1800)
        obs_stage = row['stage(m)']
        v1 = water_balance_df['volume (cu.m)'][previous_time.strftime('%Y-%m-%d %H:%M:%S')]
        v2 = water_balance_df['volume (cu.m)'][index.strftime('%Y-%m-%d %H:%M:%S')]
        if ((v1 -v2) > 14) and obs_stage < full_stage:
            water_balance_df['pumping status'][index.strftime(date_format)] = 1
            # water_balance_df['pumping (cu.m)'][index.strftime(date_format)] = 14


"""
Overflow
"""
full_vol = stage_vol_df['total_vol_cu_m'][0.66]
# print full_vol
# raise SystemExit(0)
length_check_dam = 10.4 # b
width_check_dam = 0.5 # L
# discharge_coeff = 1.704  # http://pubs.usgs.gov/wsp/0200/report.pdf page 9
no_of_contractions = 0
water_balance_df['overflow(cu.m)'] = 0.000
for index in water_balance_df.index:
    obs_stage = water_balance_df['stage(m)'][index.strftime(date_format)]
    if (np.around(obs_stage, 2)) > full_stage:
        effective_head = obs_stage - full_stage
        previous_time = index - timedelta(seconds=1800)
        if np.around(water_balance_df['stage(m)'][previous_time.strftime(date_format)], 2) > full_stage:
            water_balance_df['overflow(cu.m)'][index.strftime(date_format)] = 1800*1.84*width_check_dam*(effective_head**1.5)
        else:
            x1 = 0
            x2 = 1800
            y1 = np.around(water_balance_df['stage(m)'][previous_time.strftime(date_format)], 2)
            y2 = np.around(obs_stage, 2)
            slope = (y1 - y2)/(x1 - x2)
            intercept = y2 - (slope*x2)
            time_of_overflow = 1800 - ((full_stage-intercept)/slope)
            water_balance_df['overflow(cu.m)'][index.strftime(date_format)] = time_of_overflow*1.84*width_check_dam*(effective_head**1.5)


# for index, row in water_balance_df.iterrows():
#     # obs_stage = row['volu']
#     obs_vol = row['volume (cu.m)']
#     if obs_vol > full_vol:
#         # print obs_vol
#         water_balance_df['overflow(cu.m)'][index.strftime('%Y-%m-%d %H:%M:%S')] = obs_vol - full_vol
#         water_balance_df['volume (cu.m)'][index.strftime('%Y-%m-%d %H:%M:%S')] = full_vol
#         water_balance_df['stage(m)'][index.strftime('%Y-%m-%d %H:%M:%S')] = 1.9

# start from May 15
# water_balance_df = water_balance_df["2014-05-15":]
# print water_balance_df.head()
water_balance_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/30_min_check.csv')
# print ch_storage_df.head()

# fig = plt.figure(figsize=(11.69, 8.27))
# plt.plot_date(water_balance_df.index, water_balance_df['volume (cu.m)'], '-g')
# plt.hlines(stage_vol_df['total_vol_cu_m'][1.9], min(water_balance_df.index), max(water_balance_df.index))
# plt.title('after overflow correction')
"""
Stage vs area linear relationship
"""
stage_area_df = pd.read_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/cont_area.csv',
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
    if obs_stage >= stage_cutoff:
        x1, x2 = find_range(stage_area_df['stage_m'].tolist(), obs_stage)
        x_diff = x2-x1
        y1 = stage_area_df['total_area_sq_m'][x1]
        y2 = stage_area_df['total_area_sq_m'][x2]
        y_diff = y2 - y1
        slope = y_diff/x_diff
        y_intercept = y2 - (slope*x2)
        water_balance_df['ws_area(sq.m)'][index.strftime('%Y-%m-%d %H:%M:%S')] = (slope*obs_stage) + y_intercept
"""
Evaporation Volume estimation
"""
water_balance_df['Evaporation (cu.m)'] = (water_balance_df['Evaporation (mm/30min)'] * 0.001) * water_balance_df['ws_area(sq.m)']
# start from May 15
"""
Daily Totals of Rain, Evaporation, Overflow
"""
sum_df = water_balance_df[['Rain Collection (mm)', 'Evaporation (cu.m)', 'Evaporation (mm/30min)', 'overflow(cu.m)', 'pumping status']]
sum_df = sum_df.resample('D', how=np.sum)
# print sum_df.head(10)
"""
Daily average of Stage
"""
stage_df = water_balance_df[['stage(m)']]
stage_df = stage_df.resample('D', how=np.mean)
# print stage_df.head()
water_balance_daily_df = sum_df.join(stage_df, how='left')
water_balance_daily_df['ws_area(sq.m)'] = 0.000
for index, row in water_balance_daily_df.iterrows():
    obs_stage = row['stage(m)']  # observed stage
    x1, x2 = find_range(stage_area_df['stage_m'].tolist(), obs_stage)
    x_diff = x2-x1
    y1 = stage_area_df['total_area_sq_m'][x1]
    y2 = stage_area_df['total_area_sq_m'][x2]
    y_diff = y2 - y1
    slope = y_diff/x_diff
    y_intercept = y2 - (slope*x2)
    water_balance_daily_df['ws_area(sq.m)'][index.strftime('%Y-%m-%d %H:%M:%S')] = (slope*obs_stage) + y_intercept

# print water_balance_daily_df.head()
"""
Change in storage
"""
# separate out 23:30 readings
hour = water_balance_df.index.hour
minute = water_balance_df.index.minute
ch_storage_df = water_balance_df[['volume (cu.m)']][((hour == 23) & (minute == 30))]
ch_storage_df = ch_storage_df.resample('D', how=np.mean)
water_balance_daily_df['change_storage(cu.m)'] = 0.000
# print water_balance_daily_df.head()
check_df = ch_storage_df['2014-05-29': ] #average
print check_df.head()
for index in ch_storage_df.index:
    if index > min(ch_storage_df.index):
        previous_date = index - timedelta(days=1)
        d1_storage = ch_storage_df['volume (cu.m)'][previous_date.strftime(daily_format)]
        d2_storage = ch_storage_df['volume (cu.m)'][index.strftime(daily_format)]
        water_balance_daily_df['change_storage(cu.m)'][index.strftime(daily_format)] = d2_storage - d1_storage
# for d1, d2 in pairwise(ch_storage_df.index):
#     if d2 > d1:
#         diff = (d2-d1).days
#         if diff == 1:
#             d1_storage = ch_storage_df['volume (cu.m)'][d1.strftime('%Y-%m-%d')]
#             d2_storage = ch_storage_df['volume (cu.m)'][d2.strftime('%Y-%m-%d')]
#             water_balance_daily_df['change_storage(cu.m)'][d2.strftime('%Y-%m-%d')] = d2_storage - d1_storage
#
# print water_balance_daily_df.head()
new_df = water_balance_daily_df.join(ch_storage_df, how='right')
new_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/proof.csv')
# new_index = pd.date_range(start='2014-05-15', end='2014-09-10', freq='D' )
# print new_index
# print ch_storage_df.head()
"""
Separate out no inflow/ non rainy days
two continuous days of no rain
"""
water_balance_daily_df['status'] = "Y"
no_rain_df = water_balance_daily_df[water_balance_daily_df['Rain Collection (mm)'] == 0]
# no_rain_df['status'] = "Y"
for index in water_balance_daily_df.index:
    initial_time_stamp = min(water_balance_daily_df.index) + timedelta(days=1)
    if index > initial_time_stamp:
        start_date = index - timedelta(days=1)
        two_days_rain_df = water_balance_daily_df['Rain Collection (mm)'][start_date.strftime('%Y-%m-%d'):index.strftime('%Y-%m-%d')]
        sum_df = two_days_rain_df.sum(axis=0)
        if (sum_df == 0) and (water_balance_daily_df['change_storage(cu.m)'][index.strftime('%Y-%m-%d')] < 0) and (water_balance_daily_df['overflow(cu.m)'][index.strftime(daily_format)] == 0) and (water_balance_daily_df['pumping status'][index.strftime(daily_format)] == 0):
            water_balance_daily_df['status'][index.strftime('%Y-%m-%d')] = "N"

# print no_rain_df.head()
water_balance_daily_df.to_csv("/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/water_bal.csv")
no_rain_df.to_csv("/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/no_rain_df.csv")
dry_water_balance_df = water_balance_daily_df[water_balance_daily_df['status'] == "N"]
rain_water_balance_df = water_balance_daily_df[water_balance_daily_df['status'] == "Y"]
# dry_water_balance_df = dry_water_balance_df[dry_water_balance_df['stage(m)'] > 0.1]
# rain_water_balance_df = rain_water_balance_df[rain_water_balance_df['stage(m)'] > 0.1]
# print dry_water_balance_df.head()
# print rain_water_balance_df.head()
# # b = plot_date(dry_water_balance_df, 'change_storage(cu.m)')
"""
# Calculate infiltration
# """
# # calculate infiltration
# dry_water_balance_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/dry_wb_check.CSV')
# # print dry_water_balance_df.head()
dry_water_balance_df['infiltration(cu.m)'] = 0.000
delta_s = water_balance_daily_df['change_storage(cu.m)']
evap = water_balance_daily_df['Evaporation (cu.m)']
outflow = water_balance_daily_df['overflow(cu.m)']
# for t1, t2 in pairwise(dry_water_balance_df.index):
#     diff = abs((t2-t1).seconds)
#     if diff == 1800:
#         # print t1, t2
#         dry_water_balance_df['infiltration(cu.m)'][t1.strftime('%Y-%m-%d %H:%M:%S')] = -1*(delta_s[t2.strftime('%Y-%m-%d %H:%M:%S')] + evap[t2.strftime('%Y-%m-%d %H:%M:%S')] + outflow[t2.strftime('%Y-%m-%d %H:%M:%S')])
for index, row in dry_water_balance_df.iterrows():
    if index > min(water_balance_daily_df.index):
        t_1 = index - timedelta(days=1)
        if t_1 < max(water_balance_daily_df.index):
            dry_water_balance_df['infiltration(cu.m)'][index.strftime(daily_format)] = -1.0*(delta_s[index.strftime(daily_format)] + evap[index.strftime(daily_format)])
    # print row
dry_water_balance_df['infiltration rate (m/day)'] = 0.0

for i in dry_water_balance_df.index:
    i_1 = i - timedelta(days=1)
    dry_water_balance_df['infiltration rate (m/day)'][i.strftime(daily_format)] = dry_water_balance_df['infiltration(cu.m)'][i.strftime(daily_format)] / water_balance_daily_df['ws_area(sq.m)'][i.strftime(daily_format)]
# dry_water_balance_df['infiltration(cu.m)'] = -1.0*(evap + outflow + delta_s)
# # print dry_water_balance_df.head()
# # fig = plt.figure(figsize=(11.69, 8.27))
# # plt.plot(dry_water_balance_df['average_stage_m'], dry_water_balance_df['infiltration(cu.m)'], 'bo')
# # plt.show()
# print dry_water_balance_df
"""
Dry infiltration vs rainfall

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(11.69, 8.27))
# fig.subplots_adjust(right=0.8)
line1 = ax1.bar(water_balance_daily_df.index, water_balance_daily_df['Rain Collection (mm)'], 0.35, label=r'Rainfall(mm)')
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
# # plt.legend(loc='upper right')
# # fig.autofmt_xdate(rotation=90)
# # fig.subplots_adjust(right=0.8)
# ax3 = ax2.twiny()
# line3 = ax3.line(water_balance_daily_df.index, water_balance_daily_df['Evaporation (cu.m)'], 0.35, 'g', alpha=0.5,  label='Evaporation (cu.m)' )
# ax3.tick_params(axis='x',
#                 which='both',
#                 top='off',
#                 bottom='off',
#                 labeltop='off')
# # ax3.xaxis.tick_bottom()
# ax3.yaxis.tick_right()
fig.autofmt_xdate(rotation=90)
# lns = line1+line3
# labs = [l.get_label() for l in lns]
# ax3.legend(lns, labs, loc='upper center', fancybox=True, ncol=3, bbox_to_anchor=(0.5, 1.15))
# ax3.set_xlim([min(dry_water_balance_df.index), max(dry_water_balance_df.index)])
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.50, 0.05, 0.3])  #first one distance from plot, second height
# cax, kw = mpl.colorbar.make_axes([ax for ax in ax1.flat()])
cbar = fig.colorbar(line2, cax=cbar_ax)
cbar.ax.set_ylabel('Stage (m)')
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_591/new_dry_rain_infiltration_stage_591_30min')
# plt.show()
"""
"""
Fitting exponential function
"""
stage_cal = dry_water_balance_df['stage(m)']
# stage_cal = dry_water_balance_df['average_stage_m']
inf_cal = dry_water_balance_df['infiltration rate (m/day)']
print dry_water_balance_df
raise SystemExit(0)
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(dry_water_balance_df['stage(m)'], dry_water_balance_df['infiltration(cu.m)'], 'bo')
fig.autofmt_xdate(rotation=90)
# plt.show()
# plt.plot(stage_cal, inf_cal, 'bo', label=r'Observation')
# plt.vlines(1.9, 0, max(inf_cal), 'g')
# plt.hlines(0, min(stage_cal), max(stage_cal), 'y')
# plt.legend(loc='upper left')
# plt.xlabel(r'\textbf{Stage} (m)')
# plt.ylabel(r'\textbf{Infiltration} ($m^3/day$)')
# plt.title(r"Stage - Infiltration relationship during no inflow for 591 check dam")
# plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_591/infiltration_stage_591_30min')
# plt.show()

def func(h, alpha, beta):
    return alpha*(h**beta)

popt, pcov = curve_fit(func, stage_cal, inf_cal, maxfev=6000)

# print popt
# print pcov
# # print np.diag(pcov)
# print np.sqrt(np.diag(pcov))

# plot
stage_cal_new = np.linspace(min(stage_cal), max(stage_cal), 50)
inf_cal_new = func(stage_cal_new, *popt)
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(stage_cal, inf_cal, 'bo', label=r'Observation')
plt.plot(stage_cal_new, inf_cal_new, 'r-', label='Prediction')
plt.vlines(1.9, 0, max(inf_cal), 'g')
plt.hlines(0, min(stage_cal), max(stage_cal), 'y')
plt.legend(loc='upper right')
plt.xlabel(r'\textbf{Stage} (m)')
plt.ylabel(r'\textbf{Infiltration Rate} ($m/day$)')
plt.title(r"No inflow day's stage - infiltration relationship for 591 check dam")
plt.text(x=0.4, y=.003, fontsize=15, s=r'$Infiltration = {0:.2f}{{h_{{av}}}}^{{{1:.2f}}}$'.format(popt[0], popt[1]))
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_463/new_stage_inf_exp_dry_463_30min')
# plt.show()
# print dry_water_balance_df
# print dry_water_balance_df[dry_water_balance_df['infiltration(cu.m)'] < 0]
# plot rainfall vs stage
#
# fig, ax1 = plt.subplots(figsize=(11.69, 8.27))
# ax1.bar(water_balance_daily_df.index, water_balance_daily_df['Rain Collection (mm)'], 0.35, color='b', label=r'Rainfall(mm)')
# plt.gca().invert_yaxis()
# for t1 in ax1.get_yticklabels():
#     t1.set_color('b')
# ax1.set_ylabel('Rainfall(mm)')
# plt.legend(loc='upper left')
# ax2 = ax1.twinx()
# ax2.plot_date(water_balance_daily_df.index, water_balance_daily_df['stage(m)'], 'r', label='stage (m)')
# for t1 in ax2.get_yticklabels():
#     t1.set_color('r')
# plt.legend(loc='upper right')
# fig.autofmt_xdate(rotation=90)
# plt.show()

"""
Rainy day infiltration
"""
rain_water_balance_df['infiltration(cu.m)'] = 0.0
rain_water_balance_df['infiltration rate (m/day)'] = 0.000
for i in rain_water_balance_df.index:
    if rain_water_balance_df['stage(m)'][i.strftime(daily_format)] > 0.1:
        rain_water_balance_df['infiltration rate (m/day)'][i.strftime(daily_format)] = popt[0]*(rain_water_balance_df['stage(m)'][i.strftime(daily_format)]**popt[1])
for i in rain_water_balance_df.index:
    if rain_water_balance_df['stage(m)'][i.strftime(daily_format)] > 0.1:
        if i > min(water_balance_daily_df.index):
            i_1 = i - timedelta(days=1)
            rain_water_balance_df['infiltration(cu.m)'][i.strftime(daily_format)] = rain_water_balance_df['infiltration rate (m/day)'][i.strftime(daily_format)]*water_balance_daily_df['ws_area(sq.m)'][i.strftime(daily_format)]
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(rain_water_balance_df['stage(m)'], rain_water_balance_df['infiltration(cu.m)'], 'bo', label='Predicted Infiltration' )
# # plt.vlines(1.9, 0, 100, 'g')
# # plt.xlim([-1, 2.0])
# # plt.legend(loc='upper left')
plt.xlabel(r'\textbf{Stage} (m)')
plt.ylabel(r'\textbf{Infiltration} ($m^3/day$)')
plt.title(r"Inflow day's stage - infiltration relationship for 591 check dam")
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_463/new_rain_inf_463_30min')

"""
Pumping calculation
"""

merged_water_balance = pd.concat([dry_water_balance_df, rain_water_balance_df])
pump_df = merged_water_balance[merged_water_balance['pumping status'] > 0]
# print pump_df
merged_water_balance['pumping (cu.m)'] = 0.000
for index in pump_df.index:
    merged_water_balance['pumping (cu.m)'][index.strftime(daily_format)] = -1.0*(merged_water_balance['change_storage(cu.m)'][index.strftime(daily_format)] +
                                                                                 merged_water_balance['infiltration(cu.m)'][index.strftime(daily_format)] +
                                                                                 merged_water_balance['Evaporation (cu.m)'][index.strftime(daily_format)] +
                                                                                 merged_water_balance['overflow(cu.m)'][index.strftime(daily_format)])
# raise SystemExit(0)
"""
Inflow calculation
"""
# merged_water_balance = pd.concat([dry_water_balance_df, rain_water_balance_df])
merged_water_balance['Inflow (cu.m)'] = 0.000
delta_s_rain = water_balance_daily_df['change_storage(cu.m)']
inf_rain = merged_water_balance['infiltration(cu.m)']
evap_rain = water_balance_daily_df['Evaporation (cu.m)']
outflow_rain = water_balance_daily_df['overflow(cu.m)']
pump_rain = merged_water_balance['pumping (cu.m)']
for i, row in merged_water_balance.iterrows():
    if i > min(merged_water_balance.index):
        string1 = intern(row['status'])
        string2 = intern('N')
        if string1 != string2:
            # i_1 = i - timedelta(days=1)
            merged_water_balance['Inflow (cu.m)'][i.strftime(daily_format)] = (delta_s_rain[i.strftime(daily_format)] +
                                                                               inf_rain[i.strftime(daily_format)] +
                                                                               evap_rain[i.strftime(daily_format)] +
                                                                               outflow_rain[i.strftime(daily_format)] +
                                                                                  pump_rain[i.strftime(daily_format)])


# merged_water_balance = pd.concat([dry_water_balance_df, rain_water_balance_df])
# pd.PeriodIndex(ch_storage_df.index, freq='D')
# merged_water_balance = merged_water_balance.join(ch_storage_df, how='left')
merged_water_balance.sort_index(inplace=True)
dry_water_balance_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/new_dry_wb_30min.CSV')
rain_water_balance_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/new_rain_wb_30min.CSV')
merged_water_balance.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/new_wb_30min.CSV')
# print merged_water_balance.head()
daily_df = check_df.join(merged_water_balance, how='left')
# print daily_df.head()
daily_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/daily_3day_check.CSV')
# check_volume_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/30min_3day_check.CSV')
"""
Evaporation vs infiltration
"""
fig, ax1 = plt.subplots(figsize=(11.69, 8.27))
line1 = ax1.bar(merged_water_balance.index, merged_water_balance['Evaporation (cu.m)'], 0.45, color='r', label=r"\textbf{Evaporation ($m^3/day$)}")
# plt.title("Evaporation vs Infiltration for Check dam 591")
for t1 in ax1.get_yticklabels():
    t1.set_color('r')
ax2 = ax1.twiny()
line2 = ax2.bar(merged_water_balance.index, merged_water_balance['infiltration(cu.m)'], 0.45, color='g', alpha=0.5, label=r"\textbf{Infiltration ($m^3/day$}")
for t1 in ax2.get_yticklabels():
    t1.set_color('g')
lns = [line1, line2]
lab = [r"\textbf{Evaporation ($m^3/day$)}", r"\textbf{Infiltration ($m^3/day$}" ]
# ax2.legend(lns, lab, loc='upper center', fancybox=True, ncol=2, bbox_to_anchor=(0.5, 1.15))
fig.autofmt_xdate(rotation=90)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_463/evap_infilt_463_30min_n')
# plt.show()
# print merged_water_balance.head()
print merged_water_balance['Evaporation (cu.m)'].sum()
print merged_water_balance['infiltration(cu.m)'].sum()
print merged_water_balance['overflow(cu.m)'].sum()
print merged_water_balance['Inflow (cu.m)'].sum()
print merged_water_balance['pumping (cu.m)'].sum()

wb = (merged_water_balance['Evaporation (cu.m)'].sum()+
      merged_water_balance['infiltration(cu.m)'].sum() +
      merged_water_balance['overflow(cu.m)'].sum()+merged_water_balance['pumping (cu.m)'].sum()) - merged_water_balance['Inflow (cu.m)'].sum()
# print wb
merged_water_balance.index.name = 'Date'
print merged_water_balance.head()
merged_water_balance.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/et_infilt_463_w_of.csv')