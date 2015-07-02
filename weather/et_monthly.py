__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import checkdam.checkdam as cd

date_format = '%Y-%m-%d %H:%M:%S'
daily_format = '%Y-%m-%d'

# Weather file
weather_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/ksndmc_weather.csv'
# Rain file
rain_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/ksndmc_rain.csv'

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
z = 799
p = (1 - (2.25577 * (10 ** -5) * z))
air_p_pa = 101325 * (p ** 5.25588)
# give air pressure value
weather_df['AirPr(Pa)'] = air_p_pa
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
Radiation unit conversion
"""
weather_df['Solar Radiation (MJ/m2/30min)'] = (weather_df['Solar Radiation (Wpm2)'] * 1800) / (10 ** 6)
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
print  weather_df.head(10)
print weather_df.tail(10)