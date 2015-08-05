__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import checkdam.evaplib as evap
import checkdam.meteolib as met
from Pysolar import solar
import datetime

noyyal_evap_file = "/home/kiruba/PycharmProjects/area_of_curve/hydrology/hydrology/weather/weather_evap.csv"
noyyal_evap_df = pd.read_csv(noyyal_evap_file)


noyyal_evap_df['Date'] = pd.to_datetime(noyyal_evap_df['Date'], format="%m/%d/%Y")

noyyal_evap_df.set_index(noyyal_evap_df['Date'], inplace=True)

# print noyyal_evap_df.loc["2014-04-01", :]
# sort based on index
noyyal_evap_df.sort_index(inplace=True)
# remove duplicates
# create a new column with name 'index' and then assign corresponding date time index to it
noyyal_evap_df['index'] = noyyal_evap_df.index
print noyyal_evap_df.head()
noyyal_evap_df.drop_duplicates(subset='index', take_last=True, inplace=True)
del noyyal_evap_df['index']

noyyal_evap_df.sort_index(inplace=True)
# this gives unique value of column
# print noyyal_evap_df['Min_temp_c'].unique()
# to select all the values in df which satisfies min temp > 23.0
# print noyyal_evap_df.loc[noyyal_evap_df['Min_temp_c'] > 23.0]
# elevation metre
z = 411
# create new column
noyyal_evap_df['Air_pressure(Pa)'] = noyyal_evap_df['Atmospheric Pressure (hpa)'] * 100.0

lat = 11.0183
lon = 76.9725

noyyal_evap_df['Rext (J/m2/day)'] = 0.000
noyyal_evap_df['sunshine_hours'] = 0.0
for i in noyyal_evap_df.index:
    # print i
    # doy = solar.GetDayOfYear(i)()
    doy = int(i.strftime('%j'))
    # doy = (i - datetime.datetime(i.year, 1, 1)).days + 1
    # print doy
    sunshine_hours, rext = met.sun_NR(doy=doy, lat=lat)
    # print sunshine_hours, rext
    print noyyal_evap_df.loc[i,'Rext (J/m2/day)']
    # we are assigning the rext value to the i th date
    noyyal_evap_df.loc[i,'Rext (J/m2/day)'] = rext
    print noyyal_evap_df.loc[i,'Rext (J/m2/day)']
    noyyal_evap_df.loc[i,'sunshine_hours'] = sunshine_hours

print noyyal_evap_df.head()

noyyal_evap_df['wind_speed_m_s'] = noyyal_evap_df['Wind Speed(Kmph)'] * 0.277778
noyyal_evap_df['solar_radiation_j_sqm'] = noyyal_evap_df['Solar Radiation']*(0.041868)
noyyal_evap_df['average_temp_c'] = 0.5 * (noyyal_evap_df['Max_temp_C'] + noyyal_evap_df['Min_temp_c'])

airtemp = noyyal_evap_df['average_temp_c']
hum = noyyal_evap_df['Relative Humidity(%)']
airpress = noyyal_evap_df['Air_pressure(Pa)']
rs = noyyal_evap_df['solar_radiation_j_sqm']
rext = noyyal_evap_df['Rext (J/m2/day)']
sunshine = noyyal_evap_df['sunshine_hours']
wind_speed = noyyal_evap_df['wind_speed_m_s']

noyyal_evap_df['evaporation_mm_day'] = evap.E0(airtemp=airtemp,rh=hum, airpress=airpress, Rs=rs, Rext=rext, u=wind_speed, Z=z )
noyyal_evap_df.to_csv('/home/kiruba/PycharmProjects/area_of_curve/hydrology/hydrology/weather/noyyal_evap.csv')
print noyyal_evap_df.head()

