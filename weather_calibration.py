__author__ = 'kiruba'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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

# remove unneccessary columns
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



#plot weather parameters
# print weather_df.head()
#windspeed
# fig = plt.figure(figsize=(11.69, 8.27))
# plt.plot_date(weather_df.index, weather_df['Wind Speed (kmph)'], 'r-', label='Wind Speed (Kmph)')
# plt.ylabel(r'\textbf{Wind Speed}($Km/h$)')
# plt.title(r"Wind Speed - Aralumallige", fontsize=16)
# plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/wind_speed_aral')
# fig.autofmt_xdate(rotation=90)
#Min Air Temperature
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_df.index, weather_df["Min Air Temperature (C)"], 'r-', label='Min Air Temperature (C)')
plt.ylabel(r'\textbf{Temperature}($^\circ$C)')
plt.title(r"Minimum Temperature($^\circ$C) - Aralumallige", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/sm_min_temp')
fig.autofmt_xdate(rotation=90)

fig = plt.figure(figsize=(11.69, 8.27))
plt.plot_date(weather_df.index, weather_df["Max Air Temperature (C)"], 'r-', label='Air Temperature (C)')
plt.ylabel(r'\textbf{Temperature}($^\circ$C)')
plt.title(r"Maximum Temperature($^\circ$C) - Aralumallige", fontsize=16)
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/sm_max_temp')
fig.autofmt_xdate(rotation=90)

#humidity
# fig = plt.figure(figsize=(11.69, 8.27))
# plt.plot_date(weather_df.index, weather_df['Humidity (%)'], 'r-', label='Humidity (%)')
# plt.ylabel(r'\textbf{Humidity}(\%)')
# plt.title(r"Humidity - Aralumallige", fontsize=16)
# plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/humidity')
# fig.autofmt_xdate(rotation=90)
# #solar radiation
# fig = plt.figure(figsize=(11.69, 8.27))
# plt.plot_date(weather_df.index, weather_df['Solar Radiation (W/mm2)'], 'r-', label='Solar Radiation (W/mm2)')
# plt.ylabel(r'\textbf{Solar Radiation ($W/mm2$)')
# plt.title(r"Solar Radiation - Aralumallige", fontsize=16)
# plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/solar_rad')
# fig.autofmt_xdate(rotation=90)
# plt.show()

#calibration
wrong_temp = weather_df[weather_df['Min Air Temperature (C)'] < 0]
# print wrong_temp.index
for date_time in wrong_temp.index:
    print date_time