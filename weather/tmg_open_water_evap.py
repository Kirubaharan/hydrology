__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from bokeh.plotting import figure, show, output_file, gridplot
import checkdam.checkdam as cd
import checkdam.evaplib as evap

date_format = '%d/%m/%y %H:%M:%S'
daily_format = '%d/%m/%y %H:%M'
# raise SystemExit(0)
# hadonahalli weather station
weather_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/had_daily_tmg_may_14_feb_16.csv'
weather_df = pd.read_csv(weather_file, sep=',')
print(weather_df.head())

"""
Open water evaporation
"""
z = 820  # elevation at 77.5433, 13.3556 from srtm 30m dem
p = (1 - (2.25577 * (10 ** -5) * z))
air_p_pa = 101325 * (p ** 5.25588)
# give air pressure value
weather_df['AirPr(Pa)'] = air_p_pa

airtemp = weather_df['Air Temperature (C)']
hum = weather_df['Humidity (%)']
airpress = weather_df['AirPr(Pa)']
rs = weather_df['Estimated Solar Radiation (MJ/m2/d)'] * (10 ** 6)
rext = weather_df['Rext (MJ/m2/d)'] * (10 ** 6)
wind_speed = weather_df['Wind Speed (kmph)'] * 0.27778

weather_df.loc[:, "Evaporation (mm/day)"] = evap.E0(airtemp=airtemp, rh=hum, airpress=airpress, Rs=rs, Rext=rext, u=wind_speed, Z=z)

weather_df.to_csv('/media/kiruba/New Volume/milli_watershed/tmg_lake_water_balance/tmg_open_water_evap.csv')
print(weather_df.head())