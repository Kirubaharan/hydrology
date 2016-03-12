__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from bokeh.plotting import figure, show, output_file, gridplot
import checkdam.checkdam as cd
import scipy.stats as stats
import pymc as pm
from pymc import Normal, TruncatedNormal, deterministic
# from scipy.signal import lombscargle
from scipy import fft, arange

# no_of_iterations = 400000
# burn = 390000
# thin = 10

# for test
no_of_iterations = 1000
burn = 900
thin = 2


def mscatter(p, x, y, marker):
    p.scatter(x, y, marker=marker, size=15,
              line_color="navy", fill_color="orange", alpha=0.5)

def mtext(p, x, y, text):
    p.text(x, y, text=[text],
           text_color="firebrick", text_align="center", text_font_size="10pt")


date_format = '%d/%m/%y %H:%M:%S'
daily_format = '%d/%m/%y %H:%M'
# raise SystemExit(0)
# hadonahalli weather station
weather_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/had_may_14_feb_16.csv'
weather_df = pd.read_csv(weather_file, sep='\t', header=0, encoding='utf-8')
# raise SystemExit(0)
weather_df.columns.values[6] = 'Air Temperature (C)'
weather_df.columns.values[7] = 'Min Air Temperature (C)'
weather_df.columns.values[8] = 'Max Air Temperature (C)'
weather_df.columns.values[15] = 'Canopy Temperature (C)'
# raise SystemExit(0)
weather_df['date_time'] = pd.to_datetime(weather_df['Date'] + ' ' + weather_df['Time'], format=date_format)

weather_df = weather_df.set_index(weather_df['date_time'])
weather_df.drop(['Date',
                 'Time',
                 "date_time",
                 'Rain Collection (mm)',
                 'Barometric Pressure (KPa)',
                 'Soil Moisture',
                 'Leaf Wetness',
                 'Canopy Temperature (C)',
                 'Evapotranspiration',
                 'Charging status',
                 'Solar panel voltage',
                 'Network strength',
                 'Battery strength'], axis=1, inplace=True)

"""
Remove Duplicates
"""
# # print df_base.count()

# print df_base.head()
# print df_base.count()

weather_df['index'] = weather_df.index
weather_df.drop_duplicates(subset='index', keep='last', inplace=True)
del weather_df['index']

weather_df.sort_index(inplace=True)
"""
round seconds
"""
rounded_index = np.array(weather_df.index, dtype='datetime64[m]')
weather_df.set_index(rounded_index,inplace=True)
# raise SystemExit(0)



weather_df.loc[(weather_df['Humidity (%)'] < 12) | (weather_df['Min Air Temperature (C)'] < 10) | (weather_df['Min Air Temperature (C)'] > 36) | (weather_df['Max Air Temperature (C)'] > 36) | (weather_df['Wind Speed (kmph)'] == 0.0) | (weather_df['Wind Speed (kmph)'] > 12), 'Wind Speed (kmph)'] = np.nan
weather_df.loc[(weather_df['Humidity (%)'] < 12) | (weather_df['Min Air Temperature (C)'] < 10) | (weather_df['Min Air Temperature (C)'] > 36) | (weather_df['Max Air Temperature (C)'] > 36) | (weather_df['Wind Speed (kmph)'] == 0.0) | (weather_df['Wind Speed (kmph)'] > 12), 'Max Air Temperature (C)'] = np.nan
weather_df.loc[(weather_df['Humidity (%)'] < 12) | (weather_df['Min Air Temperature (C)'] < 10) | (weather_df['Min Air Temperature (C)'] > 36) |(weather_df['Max Air Temperature (C)'] > 36) | (weather_df['Wind Speed (kmph)'] == 0.0) | (weather_df['Wind Speed (kmph)'] > 12), 'Min Air Temperature (C)'] = np.nan
weather_df.loc[(weather_df['Humidity (%)'] < 12) | (weather_df['Min Air Temperature (C)'] < 10) | (weather_df['Min Air Temperature (C)'] > 36) | (weather_df['Max Air Temperature (C)'] > 36) | (weather_df['Wind Speed (kmph)'] == 0.0) | (weather_df['Wind Speed (kmph)'] > 12), 'Humidity (%)'] = np.nan
weather_df.loc[(weather_df['Humidity (%)'] < 12) | (weather_df['Min Air Temperature (C)'] < 10) | (weather_df['Min Air Temperature (C)'] > 36) | (weather_df['Max Air Temperature (C)'] > 36) | (weather_df['Wind Speed (kmph)'] == 0.0) | (weather_df['Wind Speed (kmph)'] > 12), 'Solar Radiation (Wpm2)'] = np.nan
# weather_df.loc[weather_df['Humidity (%)'] < 12, 'Humidity (%)'] = np.nan
weather_df.loc[:, 'Air Temperature (C)'] = 0.5*(weather_df.loc[:, 'Max Air Temperature (C)'] + weather_df.loc[:,'Min Air Temperature (C)'])


#  fill na values when weather station is giving wrong values esp between jan 26 to feb 12
weather_df.loc['2015-01-26':'2015-02-12', 'Air Temperature (C)'] = np.nan
weather_df.loc['2015-01-26':'2015-02-12', 'Wind Speed (kmph)'] = np.nan
weather_df.loc['2014-06-28':'2014-07-26', 'Wind Speed (kmph)'] = np.nan
weather_df.loc['2015-01-26':, 'Solar Radiation (Wpm2)'] = np.nan
weather_df.loc['2015-01-26':'2015-02-12', 'Humidity (%)'] = np.nan
# resample and interpolate
weather_df = weather_df.resample('30Min', how=np.mean, label='right', closed='right')
start_time = min(weather_df.index)
end_time = max(weather_df.index)
new_index = pd.date_range(start=start_time, end=end_time, freq='30Min')
weather_df = weather_df.reindex(new_index, method=None)

weather_ksndmc = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tubgere_weather_corrected_wind_speed.csv'
weather_ksndmc_df = pd.read_csv(weather_ksndmc, sep=',')
# weather_ksndmc_df.drop(['Sl no', 'HOBLI'], inplace=True, axis=1)
weather_date_format = "%Y-%m-%d %H:%M:%S"
weather_ksndmc_df['Date_Time'] = pd.to_datetime(weather_ksndmc_df['Date_Time'], format=weather_date_format)
weather_ksndmc_df.set_index(weather_ksndmc_df['Date_Time'], inplace=True)
weather_ksndmc_df.sort_index(inplace=True)
weather_ksndmc_df.resample('30Min', how=np.mean, label='right', closed='right')
# take values from ksndmc where data is misisng
weather_df.loc['2015-01-14 13:30:00': '2015-02-10 11:30:00', 'Air Temperature (C)'] = weather_ksndmc_df.loc['2015-01-14 13:30:00': '2015-02-10 11:30:00', 'TEMPERATURE']
weather_df.loc[:'2015-02-09 23:30:00', 'Wind Speed (kmph)'] = weather_ksndmc_df.loc[:'2015-02-09 23:30:00', 'WIND_SPEED']*3.6   #m/s to kmph conversion
weather_df.loc['2015-01-14 13:30:00': '2015-02-10 11:30:00', 'Humidity (%)'] = weather_ksndmc_df.loc['2015-01-14 13:30:00': '2015-02-10 11:30:00', 'HUMIDITY']
# replace ksndmc's null value (50 ) with NAN
weather_df.loc[(weather_df['Air Temperature (C)'] == 50), 'Air Temperature (C)'] = np.nan
weather_df.loc[(weather_df['Humidity (%)'] == 50), 'Humidity (%)'] = np.nan
# do interpolation only where only two continous values are missing
weather_df.interpolate(method='time', limit=2, inplace=True)
#  fill na values when weather station is giving wrong values
# weather_ksndmc_df.loc[weather_ksndmc_df]
# weather_df.to_csv('/media/kiruba/New Volume/milli_watershed/cumulative impacts/tmg_lake/weather.csv')
# #
# p_1 = figure(x_axis_type='datetime', title='Temperature (C)')
# p_1.line(weather_df.index, weather_df['Air Temperature (C)'], color='chartreuse', alpha=0.5)
# p_2 = figure(x_axis_type='datetime', title='Wind speed (kmph)')
# p_2.line(weather_df.index, weather_df['Wind Speed (kmph)'], color='crimson', alpha=0.5)
# p_3 = figure(x_axis_type='datetime', title='Solar Radiation (Wpm2)')
# p_3.line(weather_df.index, weather_df['Solar Radiation (Wpm2)'], color='darkred', alpha=0.5)
# p_4 = figure(x_axis_type='datetime', title='Humidity (%)')
# p_4.line(weather_df.index, weather_df['Humidity (%)'], color='Lime', alpha=0.5)
# p = gridplot([[p_1, p_2], [p_3, p_4]])
# output_file('/media/kiruba/New Volume/milli_watershed/cumulative impacts/tmg_lake/weather.html')
# show(p)

# round values and convert to integer
# weather_df.loc[:, 'Air Temperature (C)'] = weather_df['Air Temperature (C)'].round(decimals=1)*10
# weather_df.loc[:, 'Wind Speed (kmph)'] = weather_df['Wind Speed (kmph)'].round(decimals=1)*10

# TODO Fast Fourier Transform

hour = weather_df.index.hour
minute = weather_df.index.minute
month = weather_df.index.month
weather_df.loc[:, 'hour'] = hour
weather_df.loc[:, 'minute'] = minute
weather_df.loc[:, 'month'] = month
day_selector = ((06 <= hour) & (hour <= 18))
night_selector = ((05 >= hour) | (hour >= 19))

weather_data_day = weather_df[day_selector]
weather_data_night = weather_df[night_selector]
# print weather_data_day['Air Temperature (C)'].values
# # temp day-normal
day_temp = weather_data_day['Air Temperature (C)']
night_temp = weather_data_night['Air Temperature (C)']
hist_t_day, edges_t_day = np.histogram(weather_data_day['Air Temperature (C)'], bins=xrange(10, 40, 3), density=True)
hist_t_night, edges_t_night = np.histogram(weather_data_night['Air Temperature (C)'], bins=xrange(10, 40, 3), density=True)

mean_t_day = np.mean(weather_data_day['Air Temperature (C)'])
sigma_t_day = np.std(weather_data_day['Air Temperature (C)'])
variance_t_day = sigma_t_day**2
tau_t_day = 1.0 / variance_t_day
x_t_day = np.linspace(min(day_temp), max(day_temp), 500)

mean_t_night = np.mean(weather_data_night['Air Temperature (C)'])
sigma_t_night = np.std(weather_data_night['Air Temperature (C)'])
variance_t_night = sigma_t_night**2
tau_t_night = 1.0 / variance_t_night
x_t_night = np.linspace(min(night_temp), max(night_temp), 500)

# wind speed
hist_ws_day, edges_ws_day = np.histogram(weather_data_day['Wind Speed (kmph)'], bins=xrange(0, 14, 2))
hist_ws_night, edges_ws_night = np.histogram(weather_data_night['Wind Speed (kmph)'], bins=xrange(0, 14, 2))
# solar radiation
hist_sr_day, edges_sr_day = np.histogram(weather_data_day['Solar Radiation (Wpm2)'], bins=xrange(0, 1600, 200), density=True)
hist_sr_night, edges_sr_night = np.histogram(weather_data_night['Solar Radiation (Wpm2)'], bins=xrange(0, 1600, 200), density=True)
# humidity
hist_hum_day, edges_hum_day = np.histogram(weather_data_day['Humidity (%)'], bins=xrange(0, 110, 10), density=True)
hist_hum_night, edges_hum_night = np.histogram(weather_data_night['Humidity (%)'], bins=xrange(0, 110, 10), density=True)

# groupby halfhour
half_hour_grouped = weather_df.groupby(['month', 'hour', 'minute'])['Air Temperature (C)']
mean_t_half_hour_grouped = half_hour_grouped.mean()
#  TODO convert lambda to function http://stackoverflow.com/a/6243630/2632856
f = lambda x: x.fillna(x.mean())
half_hour_t_transformed = half_hour_grouped.transform(f)
# print half_hour_t_transformed.head()
# x_time = pd.date_range("00:00","23:30", freq="30min").time
# fig= plt.figure()
# plt.plot(x_time, mean_t_half_hour_grouped/10, 'ro-')
# plt.show()
# replace missing values
weather_df.loc[:, 'Air Temperature (C)'] = half_hour_t_transformed
# windspeed
ws_grouped = weather_df.groupby(['month', 'hour', 'minute'])["Wind Speed (kmph)"]
mean_ws_grouped = ws_grouped.mean()
ws_transformed = ws_grouped.transform(f)
weather_df.loc[:, 'Wind Speed (kmph)'] = ws_transformed
# # solar radiation
# sr_grouped = weather_df.groupby(['month', 'hour', 'minute'])["Solar Radiation (Wpm2)"]
# mean_sr_grouped = sr_grouped.mean()
# sr_transformed = sr_grouped.transform(f)
# weather_df.loc[:, 'Solar Radiation (Wpm2)'] = sr_transformed
# humidity
hum_grouped = weather_df.groupby(['month', 'hour', 'minute'])["Humidity (%)"]
mean_hum_grouped = hum_grouped.mean()
hum_transformed = hum_grouped.transform(f)
weather_df.loc[:, 'Humidity (%)'] = hum_transformed

# solar radiation
# INTERPOLATE  max and min temperature
# min temp
min_temp_grouped = weather_df.groupby(['month', 'hour', 'minute'])["Min Air Temperature (C)"]
mean_min_temp_grouped = min_temp_grouped.mean()
min_temp_transformed = min_temp_grouped.transform(f)
weather_df.loc[:, 'Min Air Temperature (C)'] = min_temp_transformed
# max temp
max_temp_grouped = weather_df.groupby(['month', 'hour', 'minute'])["Max Air Temperature (C)"]
mean_max_temp_grouped = max_temp_grouped.mean()
max_temp_transformed = max_temp_grouped.transform(f)
weather_df.loc[:, 'Max Air Temperature (C)'] = max_temp_transformed

# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
# ax1.plot(weather_df.index, weather_df['Min Air Temperature (C)'] , 'g-')
# ax2.plot(weather_df.index, weather_df['Max Air Temperature (C)'], 'r-')
# plt.show()

# http://www.fao.org/docrep/x0490e/x0490e07.htm#estimating%20missing%20climatic%20data
#  ra = krs*sqrt(tmax - tmin)*Rs  krs = 0.16 rs MJm-2time-1
"""
Half hourly Extraterrestrial Radiation Calculation(J/m2/30min)
"""
sc_default = 1367.0  # Solar constant in W/m^2 is 1367.0.
ch_591_lat = 13.260196
ch_591_long = 77.512085
weather_df['Rext (MJ/m2/30min)'] = 0.000

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)
# # ax1.plot(weather_ksndmc_df.index, weather_ksndmc_df['TEMPERATURE'], 'r-')
# # ax1.plot(weather_data_day.index, weather_data_day['corrected_temp']/10, 'g-')
# # ax1.plot(weather_data_night.index, weather_data_night['corrected_temp']/10, 'r-')
# # ax1.plot(weather_df.index, weather_df['Air Temperature'], 'g-')
# ax1.plot(weather_df.index, weather_df['Air Temperature (C)'], '-')
# # ax2.plot(weather_ksndmc_df.index, weather_ksndmc_df['WIND_SPEED']*3.6, 'r-')
# ax2.plot(weather_df.index, weather_df['Wind Speed (kmph)'], '-')
# ax3.plot(weather_df.index, weather_df['Solar Radiation (Wpm2)'], '-')
# # ax4.plot(weather_ksndmc_df.index, weather_ksndmc_df['HUMIDITY'], 'r-')
# ax4.plot(weather_df.index, weather_df['Humidity (%)'])
# ax1.set_title('Temp')
# ax2.set_title('wind speed kmph')
# ax3.set_title('Solar radiation wpm2')
# ax4.set_title("Humidity")
# plt.show()
raise SystemExit(0)
# fig = plt.figure()
# plt.plot(half_hour_t_transformed.index, half_hour_t_transformed/10, 'r-o')
# plt.plot(weather_df.index, weather_df['Air Temperature (C)'], '-bo')
# plt.show()
# raise SystemExit(0)
# # temp - day - powernorm
# temp_day = figure(title="Day Temperature", background_fill_color="#E8DDCB")
# temp_day.quad(top=hist_t_day, bottom=0, left=edges_t_day[:-1], right=edges_t_day[1:], fill_color="#036564", line_color="#033649")
# temp_day.line(x_t_day, stats.norm.pdf(x_t_day, mean_t_day, sigma_t_day),line_color="#D95B43", line_width=8, alpha=0.7)
# temp_night = figure(title="Night Temperature", background_fill_color="#E8DDCB")
# temp_night.quad(top=hist_t_night, bottom=0, left=edges_t_night[:-1], right=edges_t_night[1:], fill_color="#036564", line_color="#033649")
# temp_night.line(x_t_night, stats.norm.pdf(x_t_night, mean_t_night, sigma_t_night),line_color="#D95B43", line_width=8, alpha=0.7)
# # wind speed
# ws_day = figure(title="Day Wind Speed", background_fill_color="#E8DDCB")
# ws_day.quad(top=hist_ws_day, bottom=0, left=edges_ws_day[:-1], right=edges_ws_day[1:], fill_color="#036564", line_color="#033649")
# ws_night = figure(title="Night Wind Speed", background_fill_color="#E8DDCB")
# ws_night.quad(top=hist_ws_night, bottom=0, left=edges_ws_night[:-1], right=edges_ws_night[1:], fill_color="#036564", line_color="#033649")
# # solar radiation
# sr_day = figure(title="Day Solar Radiation", background_fill_color="#E8DDCB")
# sr_day.quad(top=hist_sr_day, bottom=0, left=edges_sr_day[:-1], right=edges_sr_day[1:], fill_color="#036564", line_color="#033649")
# sr_night = figure(title="Night Solar Radiation", background_fill_color="#E8DDCB")
# sr_night.quad(top=hist_sr_night, bottom=0, left=edges_sr_night[:-1], right=edges_sr_night[1:], fill_color="#036564", line_color="#033649")
# # humidity
# hum_day = figure(title="Day Humidity", background_fill_color="#E8DDCB")
# hum_day.quad(top=hist_hum_day, bottom=0, left=edges_hum_day[:-1], right=edges_hum_day[1:], fill_color="#036564", line_color="#033649")
# hum_night = figure(title="Night Humidity", background_fill_color="#E8DDCB")
# hum_night.quad(top=hist_hum_night, bottom=0, left=edges_hum_night[:-1], right=edges_hum_night[1:], fill_color="#036564", line_color="#033649")
#
# weather_plot = gridplot([[temp_day, temp_night], [ws_day, ws_night], [sr_day, sr_night], [hum_day, hum_night]])
# output_file('/media/kiruba/New Volume/milli_watershed/cumulative impacts/tmg_lake/day_night_weather_comp.html')
# show(weather_plot)
# raise SystemExit(0)

"""
Temperature imputation MCMC

# day
day_temp = np.where(np.isnan(day_temp), 999, day_temp)

def day_t_missing_model(dataset, mu, tau, name):
    masked_values = np.ma.masked_values(dataset, value=999)
    temp_day = Normal(name, mu, tau, value=masked_values, observed=True)
    return locals()

M_missing_day = pm.MCMC(day_t_missing_model(dataset=day_temp, mu=mean_t_day, tau=tau_t_day, name='dtemp'))
M_missing_day.sample(iter=no_of_iterations, burn=burn, thin=thin)
missing_values = np.mean(np.array(M_missing_day.trace('dtemp')[-50:-1]), axis=0)
print missing_values.dtype
masked_values = np.ma.masked_values(day_temp, value=999)
masked_values[masked_values.mask] = missing_values
weather_data_day.loc[:, 'corrected_temp'] = np.array(masked_values.data.tolist()).astype(int)
# night
night_temp = np.where(np.isnan(night_temp), 999, night_temp)

def night_t_missing_model(dataset, mu, tau, name):
    masked_values = np.ma.masked_values(dataset, value=999)
    temp_night = Normal(name, mu, tau, value=masked_values, observed=True)
    return locals()

M_missing_night = pm.MCMC(night_t_missing_model(dataset=night_temp, mu=mean_t_night, tau=tau_t_night, name='ntemp'))
M_missing_night.sample(iter=no_of_iterations, burn=burn, thin=thin)
missing_values = np.mean(np.array(M_missing_night.trace('ntemp')[-50:-1]), axis=0)
print missing_values.dtype
masked_values = np.ma.masked_values(night_temp, value=999)
masked_values[masked_values.mask] = missing_values
weather_data_night.loc[:, 'corrected_temp'] = np.array(masked_values.data.tolist()).astype(int)

# combine day and night
weather_df['Air Temperature (C)'].ix[weather_data_day.index] = weather_data_day['corrected_temp']
weather_df['Air Temperature (C)'].ix[weather_data_night.index] = weather_data_night['corrected_temp']
weather_df.sort_index(inplace=True)
"""
"""
Wind speed imputation MCMC


day_ws = weather_data_day['Wind Speed (kmph)']
night_ws = weather_data_night['Wind Speed (kmph)']
hist_ws_day, edges_ws_day = np.histogram(weather_data_day['Wind Speed (kmph)'], bins=xrange(0, 12, 2), density=True)
hist_ws_night, edges_ws_night = np.histogram(weather_data_night['Wind Speed (kmph)'], bins=xrange(0, 12, 2), density=True)

mean_ws_day = np.mean(weather_data_day['Wind Speed (kmph)'])
sigma_ws_day = np.std(weather_data_day['Wind Speed (kmph)'])
variance_ws_day = sigma_ws_day**2
tau_ws_day = 1.0 / variance_ws_day
x_ws_day = np.linspace(min(day_ws), max(day_ws), 500)

mean_ws_night = np.mean(weather_data_night['Wind Speed (kmph)'])
sigma_ws_night = np.std(weather_data_night['Wind Speed (kmph)'])
variance_ws_night = sigma_ws_night**2
tau_ws_night = 1.0 / variance_ws_night
x_ws_night = np.linspace(min(night_ws), max(night_ws), 500)

# day
# day_ws = np.where(np.isnan(day_ws), 18, day_ws)
day_ws = np.ma.masked_array(day_ws, np.isnan(day_ws))
print np.max(day_ws)


def day_ws_missing_model(dataset, mu, tau, name):
    cutoff_a = 0
    cutoff_b = 14
    masked_values = np.ma.masked_array(dataset, np.isnan(dataset))
    temp_day = TruncatedNormal(name, mu, tau, a=cutoff_a, b=cutoff_b, value=masked_values, observed=True)
    return locals()

M_missing_day = pm.MCMC(day_ws_missing_model(dataset=day_ws, mu=mean_ws_day, tau=tau_ws_day, name='dws'))
M_missing_day.sample(iter=no_of_iterations, burn=burn, thin=thin)
missing_values = np.mean(np.array(M_missing_day.trace('dws')[-50:-1]), axis=0)
print missing_values.dtype
# masked_values = np.ma.masked_values(day_ws, value=18)
masked_values = np.ma.masked_array(day_ws, np.isnan(day_ws))
masked_values[masked_values.mask] = missing_values
weather_data_day.loc[:, 'corrected_ws'] = np.array(masked_values.data.tolist()).astype(int)
# night
# night_ws = np.where(np.isnan(night_ws), 18, night_ws)

night_ws = np.ma.masked_array(night_ws, np.isnan(night_ws))

def night_ws_missing_model(dataset, mu, tau, name):
    cutoff_a = 0
    cutoff_b = 14
    print cutoff_b
    # masked_values = np.ma.masked_values(dataset, value=18)
    masked_values = np.ma.masked_array(dataset, np.isnan(dataset))
    temp_night = TruncatedNormal(name, mu, tau, a=cutoff_a, b=cutoff_b, value=masked_values, observed=True)
    return locals()

M_missing_night = pm.MCMC(night_ws_missing_model(dataset=night_ws, mu=mean_ws_night, tau=tau_ws_night, name='nws'))
M_missing_night.sample(iter=no_of_iterations, burn=burn, thin=thin)
missing_values = np.mean(np.array(M_missing_night.trace('nws')[-50:-1]), axis=0)
print missing_values.dtype
# masked_values = np.ma.masked_values(night_ws, value=18)
masked_values = np.ma.masked_array(night_ws, np.isnan(night_ws))
masked_values[masked_values.mask] = missing_values
weather_data_night.loc[:, 'corrected_ws'] = np.array(masked_values.data.tolist()).astype(int)

# combine day and night
weather_df.loc[weather_data_day.index, 'Wind Speed (kmph)'] = weather_data_day['corrected_ws']
weather_df[weather_data_night.index, 'Wind Speed (kmph)'] = weather_data_night['corrected_ws']
weather_df.sort_index(inplace=True)
"""

