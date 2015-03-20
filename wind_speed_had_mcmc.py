__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import pymc as pm
import scipy.stats as stats
from datetime import timedelta
from pymc import TruncatedNormal

"""
Weather
Read from csv and create datetime index, re arrange columns, drop unwanted columns
"""
weather_ksndmc = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tubgere_weather_01May14_10Feb15.csv'
weather_ksndmc_df = pd.read_csv(weather_ksndmc, sep=',')  # read from csv
weather_ksndmc_df.drop(['Sl no', 'HOBLI'], inplace=True, axis=1)  # drop columns
weather_date_format = "%d-%b-%y %H:%M:%S+05:30"
weather_ksndmc_df['date_time'] = pd.to_datetime(weather_ksndmc_df['DATE'] + " " + weather_ksndmc_df['TIME'], format=weather_date_format)
weather_ksndmc_df.set_index(weather_ksndmc_df['date_time'], inplace=True) # create datetime index
weather_ksndmc_df.sort_index(inplace=True)  # sort
cols = weather_ksndmc_df.columns.tolist()  # rearrange columns
cols = cols[-1:] + cols[:-1]
weather_ksndmc_df = weather_ksndmc_df[cols]
weather_ksndmc_df.drop(['date_time', 'DATE', 'TIME'], inplace=True, axis=1) # drop columns

# weather_ksndmc_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tubgere_weather_01May14_10Feb15_corrected.csv')

# fig = plt.figure()
# plt.plot(weather_regr_df['HUMIDITY'], weather_regr_df['TEMPERATURE'], 'go')
# plt.show()
# select only data with 15 minute interval
minute = weather_ksndmc_df.index.minute
weather_ksndmc_df = weather_ksndmc_df[((minute == 0) | (minute == 15) | (minute == 30) | (minute == 45) | (minute == 60))]
# drop duplicates
weather_ksndmc_df['index'] = weather_ksndmc_df.index
weather_ksndmc_df.drop_duplicates(subset='index', take_last=True, inplace=True)
del weather_ksndmc_df['index']
weather_ksndmc_df = weather_ksndmc_df.sort()

# h = weather_ksndmc_df['WIND_SPEED'][weather_ksndmc_df['WIND_SPEED'] < 3.0]
# h = sorted(h)
# fit = stats.norm.pdf(h, np.mean(h), np.std(h))
# fig = plt.figure()
# plt.plot(h, fit, '-o')
# plt.hist(h, normed=True)
# plt.show()
# raise SystemExit(0)
# multiply by 10 so that we get integer output
weather_ksndmc_df['WIND_SPEED'] = weather_ksndmc_df['WIND_SPEED']*10
start_time = min(weather_ksndmc_df.index)
end_time = max(weather_ksndmc_df.index)
new_index = pd.date_range(start=start_time, end=end_time, freq='15min')
weather_ksndmc_df = weather_ksndmc_df.reindex(new_index, fill_value=50)
# print len(weather_ksndmc_df.index)
# print np.where(weather_ksndmc_df['WIND_SPEED'] == 50)[0][50]
# consider values above 30 as missing as it is unrealistic
weather_ksndmc_df['WIND_SPEED'] = np.where(weather_ksndmc_df['WIND_SPEED'] > 30.0, None, weather_ksndmc_df['WIND_SPEED'])
# fill by interpolation if only one value is missing
max_limit = max(weather_ksndmc_df.index) - timedelta(days=1)
min_limit = min(weather_ksndmc_df.index) + timedelta(days=1)
for index in weather_ksndmc_df.index:
    if weather_ksndmc_df['WIND_SPEED'][index] == None:
        if ((index > min_limit) and (index < max_limit)):
            previous_day_value = weather_ksndmc_df['WIND_SPEED'][index - timedelta(days=1)]
            next_day_value = weather_ksndmc_df['WIND_SPEED'][index + timedelta(days=1)]
            if ((previous_day_value != None) and (next_day_value != None)):
                weather_ksndmc_df['WIND_SPEED'][index] = 0.5*(previous_day_value+next_day_value)

# missing_values = np.ma.masked_values(weather_ksndmc_df['WIND_SPEED'].values, value=None)
# # print missing_values.mask.sum()
# fig = plt.figure()
# axes = fig.add_subplot(1,1,1, axisbg='red')
# axes.plot(weather_ksndmc_df.index, weather_ksndmc_df['WIND_SPEED'], '-')
# plt.show()
#  take  wind speed values for 7 days prior to missing and 15 days after
prior_missing_data = weather_ksndmc_df['2014-08-21':'2014-08-28 17:15:00']
after_missing_data = weather_ksndmc_df['2014-09-03 14:30:00': '2014-09-10']
# check for none values in selected data, if none is present, np.mean will not work
for index in prior_missing_data.index:
    if prior_missing_data['WIND_SPEED'][index] == None:
        print index
for index in after_missing_data.index:
    if after_missing_data['WIND_SPEED'][index] == None:
        print index
full_data = weather_ksndmc_df['2014-08-21':'2014-09-10']
hour_p = prior_missing_data.index.hour
day_selector_p = ((06 <= hour_p) & (hour_p <= 18))
night_selector_p = ((05 >= hour_p) | (hour_p>=19))
prior_missing_data_day = prior_missing_data[day_selector_p]
prior_missing_data_night = prior_missing_data[night_selector_p]
hour_a = after_missing_data.index.hour
day_selector_a = ((06 <= hour_a) & (hour_a <= 18))
night_selector_a = ((05 >= hour_a) | (hour_a>=19))
after_missing_data_day = after_missing_data[day_selector_a]
after_missing_data_night = after_missing_data[night_selector_a]
day_df = pd.concat((prior_missing_data_day, after_missing_data_day))
day_wind_speed = day_df['WIND_SPEED'].values
day_mean = np.mean(day_wind_speed)
day_sigma = np.std(day_wind_speed)
day_variance = np.square(day_sigma)
night_df = pd.concat((prior_missing_data_night, after_missing_data_night))
night_wind_speed = night_df['WIND_SPEED'].values
night_mean = np.mean(night_wind_speed)
night_sigma = np.std(night_wind_speed)
night_variance = np.square(night_sigma)
print " Day Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" %(day_mean, day_sigma, day_variance)
print " Night Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" %(night_mean, night_sigma, night_variance)
day_tau = 1.0/day_variance
night_tau = 1.0/night_variance
hour_f = full_data.index.hour
day_selector_f = ((06 <= hour_f) & (hour_f <= 18))
night_selector_f = ((05 >= hour_f) | (hour_f >=19))
full_data_day = full_data[day_selector_f]
full_data_day_wind_speed = full_data_day['WIND_SPEED'].values
full_data_night = full_data[night_selector_f]
full_data_night_wind_speed = full_data_night['WIND_SPEED'].values


def day_missing_model():
    # Mean
    mu = day_mean
    # Tau
    tau = day_tau
    # first cutoff
    cutoff_a = 0
    # second cutoff
    cutoff_b = 30
    masked_values = np.ma.masked_values(full_data_day_wind_speed, value=None)
    print masked_values.mask.sum()
    wind_speed_day = TruncatedNormal('dws', mu, tau, cutoff_a, cutoff_b, value=masked_values, observed=True)
    return locals()

M_missing_day = pm.MCMC(day_missing_model())
M_missing_day.sample(iter=100000, burn=95000)
missing_values = np.mean(np.array(M_missing_day.trace('dws')[:]), axis=0)
print len(missing_values)
print len(full_data_day.index)
masked_values = np.ma.masked_values(full_data_day_wind_speed, value=None)
masked_values[masked_values.mask] = missing_values
full_data_day['corrected_wind_speed'] = masked_values.data.tolist()
print full_data_day.head()
plt.figure()
plt.plot(full_data_day.index, masked_values, 'ro', label="Missing")
plt.plot(full_data_day.index, full_data_day['WIND_SPEED'], 'go', label="Observed")
plt.title("Day Wind Speed Values")
plt.legend()
plt.show()


def night_missing_model():
    # Mean
    mu = night_mean
    # Tau
    tau = night_tau
    # first cutoff
    cutoff_a = 0
    # second cutoff
    cutoff_b = 30
    masked_values = np.ma.masked_values(full_data_night_wind_speed, value=None)
    print masked_values.mask.sum()
    wind_speed_day = TruncatedNormal('nws', mu, tau, cutoff_a, cutoff_b, value=masked_values, observed=True)
    return locals()

M_missing_night = pm.MCMC(night_missing_model())
M_missing_night.sample(iter=100000, burn=95000)
missing_values = np.mean(np.array(M_missing_night.trace('nws')[:]), axis=0)
print len(missing_values)
print len(full_data_night.index)
masked_values = np.ma.masked_values(full_data_night_wind_speed, value=None)
masked_values[masked_values.mask] = missing_values
full_data_night['corrected_wind_speed'] = masked_values.data.tolist()
print full_data_night.head()
plt.figure()
plt.plot(full_data_night.index, masked_values, 'ro', label="Missing")
plt.plot(full_data_night.index, full_data_night['WIND_SPEED'], 'go', label="Observed")
plt.title("Night Wind Speed")
plt.legend()
plt.show()
raise SystemExit(0)

fig = plt.figure()
plt.plot(prior_missing_data_day, stats.norm.pdf(prior_missing_data_day, mean_p_d, std_p_d), 'r-o', label="Day")
plt.plot(prior_missing_data_night, stats.norm.pdf(prior_missing_data_night, mean_p_n, std_p_n), 'g-o', label='Night')
plt.text(5, 0.11, "Day Mean = %0.2f and STD = %0.2f\n" %(mean_p_d, std_p_d) + "Night Mean = %0.2f and STD = %0.2f" %(mean_p_n, std_p_n), fontsize=11)
plt.legend()
plt.show()
after_missing_data = sorted(after_missing_data)
mean_a = np.mean(after_missing_data)
std_a = np.std(after_missing_data)
fit_a = stats.norm.pdf(after_missing_data, mean_a, std_a)
print "Mean = %0.2f and STD = %0.2f" %(mean_a, std_a)
fig = plt.figure()
plt.plot(after_missing_data, fit_a, '-o')
# plt.hist(after_missing_data, normed=True)
plt.show()
Z = weather_ksndmc_df['WIND_SPEED'].values
fig = plt.figure()
plt.plot(weather_ksndmc_df.index, Z, '-o')
plt.show()