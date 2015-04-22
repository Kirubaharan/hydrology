__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import pymc as pm
import scipy.stats as stats
from datetime import timedelta
from datetime import datetime
from pymc import TruncatedNormal
from pymc import Normal, Exponential
import matplotlib
# data = np.array([None, None, None, 12, 17, 20])
# masked_values = np.ma.masked_array(data, np.equal(data, None), fill_value=10)
# x = pm.TruncatedNormal('x', mu=15, tau=0.1, a=7, b=27, value=data, observed=True)
# print x.value
# raise SystemExit(0)
matplotlib.rc('font', **{'family': 'sans-serif', 'serif': ['Computer Modern Roman']})
matplotlib.rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=36)

params = {'axes.labelsize': 28, # fontsize for x and y labels (was 10)
          'axes.titlesize': 30,
          'text.fontsize': 30, # was 10
          'legend.fontsize': 30, # was 10
           'xtick.labelsize': 28,
           'ytick.labelsize': 28,
          'text.usetex': True,
          'font.family': 'serif'
          }
matplotlib.rcParams.update(params)
"""
MCMC iteration parameters
"""
no_of_iterations = 200000
burn = 190000
thin = 10

"""
Weather
Read from csv and create datetime index, re arrange columns, drop unwanted columns
"""
weather_ksndmc = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tubgere_weather_01May14_10Feb15.csv'
weather_ksndmc_df = pd.read_csv(weather_ksndmc, sep=',')  # read from csv
weather_ksndmc_df.drop(['Sl no', 'HOBLI'], inplace=True, axis=1)  # drop columns
weather_date_format = "%d-%b-%y %H:%M:%S+05:30"
weather_ksndmc_df['date_time'] = pd.to_datetime(weather_ksndmc_df['DATE'] + " " + weather_ksndmc_df['TIME'],
                                                format=weather_date_format)
weather_ksndmc_df.set_index(weather_ksndmc_df['date_time'], inplace=True)  # create datetime index
weather_ksndmc_df.sort_index(inplace=True)  # sort
cols = weather_ksndmc_df.columns.tolist()  # rearrange columns
cols = cols[-1:] + cols[:-1]
weather_ksndmc_df = weather_ksndmc_df[cols]
weather_ksndmc_df.drop(['date_time', 'DATE', 'TIME'], inplace=True, axis=1)  # drop columns

# weather_ksndmc_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tubgere_weather_01May14_10Feb15_corrected.csv')

# fig = plt.figure()
# plt.plot(weather_regr_df['HUMIDITY'], weather_regr_df['TEMPERATURE'], 'go')
# plt.show()
# select only data with 15 minute interval
minute = weather_ksndmc_df.index.minute
weather_ksndmc_df = weather_ksndmc_df[
    ((minute == 0) | (minute == 15) | (minute == 30) | (minute == 45) | (minute == 60))]
# drop duplicates
weather_ksndmc_df['index'] = weather_ksndmc_df.index
weather_ksndmc_df.drop_duplicates(subset='index', take_last=True, inplace=True)
del weather_ksndmc_df['index']
weather_ksndmc_df = weather_ksndmc_df.sort()
print len(weather_ksndmc_df.index)
# h = weather_ksndmc_df['WIND_SPEED'][weather_ksndmc_df['WIND_SPEED'] < 3.0]
# h = sorted(h)
# fit = stats.norm.pdf(h, np.mean(h), np.std(h))
# fig = plt.figure()
# plt.plot(weather_ksndmc_df.index, weather_ksndmc_df['WIND_SPEED'], '-g')
# # plt.plot(h, fit, '-o')
# # plt.hist(h, normed=True)
# plt.show()
# raise SystemExit(0)
# multiply by 10 so that we get integer output
weather_ksndmc_df['WIND_SPEED'] *= 10
start_time = min(weather_ksndmc_df.index)
end_time = max(weather_ksndmc_df.index)
new_index = pd.date_range(start=start_time, end=end_time, freq='15min')
weather_ksndmc_df = weather_ksndmc_df.reindex(new_index, fill_value=50)
print len(weather_ksndmc_df.index)
print np.where(weather_ksndmc_df['WIND_SPEED'] == 50)[0][50]
# consider values above 30 as missing as it is unrealistic
weather_ksndmc_df["WIND_SPEED"] = np.where(weather_ksndmc_df["WIND_SPEED"] > 30.0, None,
                                           weather_ksndmc_df['WIND_SPEED'])
# weather_ksndmc_df['WIND_SPEED'] = np.where(weather_ksndmc_df['WIND_SPEED'] < 5.0, 999, weather_ksndmc_df['WIND_SPEED'])
fig = plt.figure()
plt.plot(weather_ksndmc_df.index, weather_ksndmc_df['WIND_SPEED']/10, '-b')
plt.ylabel("Wind Speed (m/s)")
fig.autofmt_xdate(rotation=45)
plt.title("Tubgere Wind Speed data")
plt.show()
# raise SystemExit(0)
# fill by interpolation if only one value is missing
max_limit = max(weather_ksndmc_df.index) - timedelta(days=1)
min_limit = min(weather_ksndmc_df.index) + timedelta(days=1)
for index in weather_ksndmc_df.index:
    if weather_ksndmc_df['WIND_SPEED'][index] is None and (index > min_limit) and (index < max_limit):
        previous_day_value = weather_ksndmc_df['WIND_SPEED'][index - timedelta(days=1)]
        next_day_value = weather_ksndmc_df['WIND_SPEED'][index + timedelta(days=1)]
        if (previous_day_value != None) and (next_day_value != None):
            weather_ksndmc_df["WIND_SPEED"][index] = 0.5 * (previous_day_value + next_day_value)

missing_values = np.ma.masked_values(weather_ksndmc_df['WIND_SPEED'].values, value=None)
print missing_values.mask.sum()
# raise SystemExit(0)
# weather_ksndmc_df['WIND_SPEED'].replace([None], -5, inplace=True)
print weather_ksndmc_df['WIND_SPEED']['2014-08-28 17:30:00']
# fig = plt.figure()
# plt.plot(weather_ksndmc_df.index, weather_ksndmc_df['WIND_SPEED'], '-o')
# plt.show()
# raise SystemExit(0)
# take  wind speed values for 7 days prior to missing and 15 days after
prior_missing_data = weather_ksndmc_df['2014-08-25':'2014-08-28 17:15:00']
after_missing_data = weather_ksndmc_df['2014-09-03 14:30:00': '2014-09-06']
# check for none values in selected data, if none is present, np.mean will not work
for index in prior_missing_data.index:
    if prior_missing_data['WIND_SPEED'][index] is None:
        print index
for index in after_missing_data.index:
    if after_missing_data['WIND_SPEED'][index] is None:
        print index
full_data = weather_ksndmc_df['2014-08-25':'2014-09-06']
fig = plt.figure()
plt.plot(full_data.index, full_data['WIND_SPEED']/10.0, '-bo')
plt.ylabel("Wind Speed (m/s)")
fig.autofmt_xdate(rotation=45)
# plt.title("MCMC Imputation")
plt.show()
# raise SystemExit(0)
# missing_values = np.ma.masked_values(full_data['WIND_SPEED'].values, value=None)
# print missing_values.mask.sum()
hour_p = prior_missing_data.index.hour
day_selector_p = ((06 <= hour_p) & (hour_p <= 18))
night_selector_p = ((05 >= hour_p) | (hour_p >= 19))
prior_missing_data_day = prior_missing_data[day_selector_p]
prior_missing_data_night = prior_missing_data[night_selector_p]
hour_a = after_missing_data.index.hour
day_selector_a = ((06 <= hour_a) & (hour_a <= 18))
night_selector_a = ((05 >= hour_a) | (hour_a >= 19))
after_missing_data_day = after_missing_data[day_selector_a]
after_missing_data_night = after_missing_data[night_selector_a]
day_df = pd.concat((prior_missing_data_day, after_missing_data_day))
# day_df = day_df[day_df['WIND_SPEED'] > 5.0]
day_wind_speed = day_df['WIND_SPEED'].values
day_mean = np.mean(day_wind_speed)
day_sigma = np.std(day_wind_speed)
day_variance = day_sigma ** 2
# fig = plt.figure()
# plt.plot(sorted(day_wind_speed), stats.norm.pdf(sorted(day_wind_speed), day_mean, day_sigma))
# plt.hist(sorted(day_wind_speed), normed=True)
# plt.show()
for index in day_df.index:
    if day_df['WIND_SPEED'][index] is None:
        print index

night_df = pd.concat((prior_missing_data_night, after_missing_data_night))
night_wind_speed = night_df['WIND_SPEED'].values
night_mean = np.mean(night_wind_speed)
night_sigma = np.std(night_wind_speed)
night_variance = night_sigma ** 2
fig = plt.figure()
plt.plot(sorted(day_wind_speed), stats.norm.pdf(sorted(day_wind_speed), day_mean, day_sigma), 'g-o', label='Day')
night_fit = stats.norm.pdf(sorted(night_wind_speed), night_mean, night_sigma)
plt.plot(sorted(night_wind_speed), night_fit, '-ro', label="Night")
plt.text(5, 0.11, "Day Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f \n" % (day_mean, day_sigma, day_variance) + "Night Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" % (night_mean, night_sigma, night_variance), fontsize=11)
plt.title("PDF of wind speed for time period 2014-08-25 : 2014-09-06")
plt.legend().draggable()
plt.show()
# raise SystemExit(0)
print " Day Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" % (day_mean, day_sigma, day_variance)
print " Night Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" % (night_mean, night_sigma, night_variance)
day_tau = 1.0 / day_variance
night_tau = 1.0 / night_variance
hour_f = full_data.index.hour
day_selector_f = ((06 <= hour_f) & (hour_f <= 18))
night_selector_f = ((05 >= hour_f) | (hour_f >= 19))
full_data_day = full_data[day_selector_f]
# full_data_day = full_data_day[(full_data_day['WIND_SPEED'] > 5.0)]
# print min(full_data_day['WIND_SPEED'])daily pattern wind speed
# full_data_day['WIND_SPEED'] = np.where(full_data_day['WIND_SPEED'] == 999, None,  full_data_day['WIND_SPEED'])

# full_data_day.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tub_test.csv')
# raise SystemExit(0)
full_data_day_wind_speed = full_data_day['WIND_SPEED'].values
full_data_night = full_data[night_selector_f]
full_data_night_wind_speed = full_data_night['WIND_SPEED'].values
# fig = plt.figure()
# plt.plot(full_data_day.index, full_data_day['WIND_SPEED'], 'o')
# plt.show()
# raise SystemExit(0)


def day_missing_model():
    # Mean
    mu = day_mean
    # Tau
    tau = day_tau
    # first cutoff
    cutoff_a = 0
    print cutoff_a
    # second cutoff
    cutoff_b = 27.0
    print cutoff_b
    masked_values = np.ma.masked_values(full_data_day_wind_speed, value=None)
    print masked_values.mask.sum()
    print masked_values.data.max()
    wind_speed_day = TruncatedNormal('dws', mu, tau, a=cutoff_a, b=cutoff_b, value=masked_values, observed=True)
    return locals()


M_missing_day = pm.MCMC(day_missing_model())
M_missing_day.sample(iter=no_of_iterations, burn=burn, thin=thin)
missing_values = np.mean(np.array(M_missing_day.trace('dws')[-50:-1]), axis=0)
print missing_values.dtype
# print len(missing_values)
# print len(full_data_day.index)
# full_data_day.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tubgere_prior.csv')
masked_values = np.ma.masked_values(full_data_day_wind_speed, value=None)
masked_values[masked_values.mask] = missing_values
full_data_day['corrected_wind_speed'] = np.array(masked_values.data.tolist())
full_data_day['corrected_wind_speed'] = full_data_day['corrected_wind_speed'].astype(int)
# full_data_day.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tubgere_after.csv')

def night_missing_model():
    # Mean
    mu = night_mean
    # Tau
    tau = night_tau
    # first cutoff
    cutoff_a = 0
    # second cutoff
    cutoff_b = 22.0
    masked_values = np.ma.masked_values(full_data_night_wind_speed, value=None)
    print masked_values.mask.sum()
    wind_speed_day = TruncatedNormal('nws', mu, tau, cutoff_a, cutoff_b, value=masked_values, observed=True)
    return locals()


M_missing_night = pm.MCMC(night_missing_model())
M_missing_night.sample(iter=no_of_iterations, burn=burn, thin=thin)
missing_values = np.mean(np.array(M_missing_night.trace('nws')[-50:-1]), axis=0)
# print len(missing_values)
# print len(full_data_night.index)
masked_values = np.ma.masked_values(full_data_night_wind_speed, value=None)
masked_values[masked_values.mask] = missing_values
full_data_night['corrected_wind_speed'] = np.array(masked_values.data.tolist())
full_data_night['corrected_wind_speed'] = full_data_night['corrected_wind_speed'].astype(int)
# print full_data_night.head()
# plt.figure()
# plt.plot(full_data_night.index, masked_values, 'ro', label="Missing")
# plt.plot(full_data_night.index, full_data_night['WIND_SPEED'], 'go', label="Observed")
# plt.title("Night Wind Speed")
# plt.legend()
# plt.show()


def verify_dataframe(var):
    if isinstance(var, pd.DataFrame):
        print "Dataframe"
    elif isinstance(var, pd.Series):
        print "Series"


full_data = pd.concat((full_data_day, full_data_night))
full_data.sort_index(inplace=True)
print full_data.head()
verify_dataframe(full_data)
fig = plt.figure()
plt.plot(full_data.index, full_data['corrected_wind_speed']/10.0, '-ro', label="Estimated")
plt.plot(full_data.index, full_data['WIND_SPEED']/10.0, '-bo', label="Observed")
plt.ylabel("Wind Speed (m/s)")
plt.title("MCMC Imputation")
plt.legend().draggable()
plt.show()
raise SystemExit(0)
"""
Missing period from Jan 06 to Jan 12
"""
prior_missing_data = weather_ksndmc_df['2014-12-25':'2015-01-06 21:45:00']
after_missing_data = weather_ksndmc_df['2015-01-12 13:00:00': '2015-01-27']
# check for none values in selected data, if none is present, np.mean will not work
for index in prior_missing_data.index:
    if prior_missing_data['WIND_SPEED'][index] is None:
        print index
for index in after_missing_data.index:
    if after_missing_data['WIND_SPEED'][index] is None:
        print index
full_data_jan = weather_ksndmc_df['2014-12-25':'2015-01-27']
missing_values = np.ma.masked_values(full_data_jan['WIND_SPEED'].values, value=None)
print missing_values.mask.sum()
hour_p = prior_missing_data.index.hour
day_selector_p = ((06 <= hour_p) & (hour_p <= 18))
night_selector_p = ((05 >= hour_p) | (hour_p >= 19))
prior_missing_data_day = prior_missing_data[day_selector_p]
prior_missing_data_night = prior_missing_data[night_selector_p]
hour_a = after_missing_data.index.hour
day_selector_a = ((06 <= hour_a) & (hour_a <= 18))
night_selector_a = ((05 >= hour_a) | (hour_a >= 19))
after_missing_data_day = after_missing_data[day_selector_a]
after_missing_data_night = after_missing_data[night_selector_a]
day_df = pd.concat((prior_missing_data_day, after_missing_data_day))
# day_df = day_df[day_df['WIND_SPEED'] > 5.0]
day_wind_speed = day_df['WIND_SPEED'].values
day_mean = np.mean(day_wind_speed)
day_sigma = np.std(day_wind_speed, ddof=1)
day_variance = day_sigma ** 2
# fig = plt.figure()
# plt.plot(sorted(day_wind_speed), stats.norm.pdf(sorted(day_wind_speed), day_mean, day_sigma))
# plt.hist(sorted(day_wind_speed), normed=True)
# plt.title("Day")
# plt.show()

night_df = pd.concat((prior_missing_data_night, after_missing_data_night))
night_wind_speed = night_df['WIND_SPEED'].values
night_mean = np.mean(night_wind_speed)
night_sigma = np.std(night_wind_speed)
night_variance = night_sigma ** 2
# fig = plt.figure()
# plt.plot(sorted(night_wind_speed), stats.norm.pdf(sorted(night_wind_speed), night_mean, night_sigma))
# plt.title("Night")
# plt.show()
# fig = plt.figure()
# plt.plot(sorted(day_wind_speed), stats.norm.pdf(sorted(day_wind_speed), day_mean, day_sigma), 'g-o', label='Day')
# night_fit = stats.norm.pdf(sorted(night_wind_speed), night_mean, night_sigma)
# plt.plot(sorted(night_wind_speed), night_fit, '-ro', label="Night")
# plt.text(5, 0.11, "Day Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f \n" % (day_mean, day_sigma, day_variance) + "Night Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" % (night_mean, night_sigma, night_variance), fontsize=11)
# plt.title("PDF of wind speed for time period 2015-01-06 : 2015-01-12")
# plt.legend().draggable()
# plt.show()
print " Day Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" % (day_mean, day_sigma, day_variance)
print " Night Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" % (night_mean, night_sigma, night_variance)
day_tau = 1.0 / day_variance
night_tau = 1.0 / night_variance
hour_f = full_data_jan.index.hour
day_selector_f = ((06 <= hour_f) & (hour_f <= 18))
night_selector_f = ((05 >= hour_f) | (hour_f >= 19))
full_data_jan_day = full_data_jan[day_selector_f]
# full_data_jan_day = full_data_jan_day[(full_data_jan_day['WIND_SPEED'] > 5.0)]
# print min(full_data_jan_day['WIND_SPEED'])daily pattern wind speed
# full_data_jan_day['WIND_SPEED'] = np.where(full_data_jan_day['WIND_SPEED'] == 999, None,  full_data_jan_day['WIND_SPEED'])

# full_data_jan_day.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tub_test.csv')
# raise SystemExit(0)
full_data_jan_day_wind_speed = full_data_jan_day['WIND_SPEED'].values
full_data_jan_night = full_data_jan[night_selector_f]
full_data_jan_night_wind_speed = full_data_jan_night['WIND_SPEED'].values
# fig = plt.figure()
# plt.plot(full_data_jan_day.index, full_data_jan_day['WIND_SPEED'], 'o')
# plt.show()
# raise SystemExit(0)


def day_missing_model():
    # Mean
    mu = day_mean
    # Tau
    tau = day_tau
    # first cutoff
    cutoff_a = 0
    print cutoff_a
    # second cutoff
    cutoff_b = 22
    print cutoff_b
    masked_values = np.ma.masked_values(full_data_jan_day_wind_speed, value=None)
    print masked_values.data.min()
    print masked_values.data.max()
    wind_speed_day = TruncatedNormal('dws', mu, tau, a=cutoff_a, b=cutoff_b, value=masked_values, observed=True)
    return locals()


M_missing_day = pm.MCMC(day_missing_model())
M_missing_day.sample(iter=no_of_iterations, burn=burn, thin=thin)
missing_values = np.mean(np.array(M_missing_day.trace('dws')[-50:-1]), axis=0)
# print len(missing_values)
# print len(full_data_jan_day.index)
masked_values = np.ma.masked_values(full_data_jan_day_wind_speed, value=None)
masked_values[masked_values.mask] = missing_values
full_data_jan_day['corrected_wind_speed'] = np.array(masked_values.data.tolist())
full_data_jan_day['corrected_wind_speed'] = full_data_jan_day['corrected_wind_speed'].astype(int)
# plt.figure()
# plt.plot(full_data_jan_day.index, full_data_jan_day['corrected_wind_speed'], 'ro', label="Missing")
# plt.plot(full_data_jan_day.index, full_data_jan_day['WIND_SPEED'], 'go', label="Observed")
# plt.title("Day Wind Speed Values")
# plt.legend()
# plt.show()


def night_missing_model():
    # Mean
    mu = night_mean
    # Tau
    tau = night_tau
    # first cutoff
    cutoff_a = 0
    # second cutoff
    cutoff_b = 18
    masked_values = np.ma.masked_values(full_data_jan_night_wind_speed, value=None)
    print masked_values.mask.sum()
    wind_speed_day = TruncatedNormal('nws', mu, tau, cutoff_a, cutoff_b, value=masked_values, observed=True)
    return locals()


M_missing_night = pm.MCMC(night_missing_model())
M_missing_night.sample(iter=no_of_iterations, burn=burn, thin=thin)
missing_values = np.mean(np.array(M_missing_night.trace('nws')[-50:-1]), axis=0)
# print len(missing_values)
# print len(full_data_jan_night.index)
masked_values = np.ma.masked_values(full_data_jan_night_wind_speed, value=None)
masked_values[masked_values.mask] = missing_values
full_data_jan_night['corrected_wind_speed'] = np.array(masked_values.data.tolist())
full_data_jan_night['corrected_wind_speed'] = full_data_jan_night['corrected_wind_speed'].astype(int)
# print full_data_jan_night.head()
# plt.figure()
# plt.plot(full_data_jan_night.index, masked_values, 'ro', label="Missing")
# plt.plot(full_data_jan_night.index, full_data_jan_night['WIND_SPEED'], 'go', label="Observed")
# plt.title("Night Wind Speed")
# plt.legend()
# plt.show()
full_data_jan = pd.concat((full_data_jan_day, full_data_jan_night))
full_data_jan.sort_index(inplace=True)
verify_dataframe(full_data_jan)
# fig = plt.figure()
# plt.plot(full_data_jan.index, full_data_jan['corrected_wind_speed'], '-o')
# plt.show()
"""
Missing data July 31 to Aug 04
"""
prior_missing_data = weather_ksndmc_df['2014-07-27':'2014-07-31 17:00:00']
after_missing_data = weather_ksndmc_df['2014-08-04 13:00:00': '2014-08-08']
# check for none values in selected data, if none is present, np.mean will not work
for index in prior_missing_data.index:
    if prior_missing_data['WIND_SPEED'][index] is None:
        print index
for index in after_missing_data.index:
    if after_missing_data['WIND_SPEED'][index] is None:
        print index
full_data_jul = weather_ksndmc_df['2014-07-27':'2014-08-08']
missing_values = np.ma.masked_values(full_data_jul['WIND_SPEED'].values, value=None)
# print missing_values.mask.sum()
hour_p = prior_missing_data.index.hour
day_selector_p = ((06 <= hour_p) & (hour_p <= 18))
night_selector_p = ((05 >= hour_p) | (hour_p >= 19))
prior_missing_data_day = prior_missing_data[day_selector_p]
prior_missing_data_night = prior_missing_data[night_selector_p]
hour_a = after_missing_data.index.hour
day_selector_a = ((06 <= hour_a) & (hour_a <= 18))
night_selector_a = ((05 >= hour_a) | (hour_a >= 19))
after_missing_data_day = after_missing_data[day_selector_a]
after_missing_data_night = after_missing_data[night_selector_a]
day_df = pd.concat((prior_missing_data_day, after_missing_data_day))
# day_df = day_df[day_df['WIND_SPEED'] > 5.0]
day_wind_speed = day_df['WIND_SPEED'].values
day_mean = np.mean(day_wind_speed)
day_sigma = np.std(day_wind_speed, ddof=1)
day_variance = day_sigma ** 2
# fig = plt.figure()
# plt.plot(sorted(day_wind_speed), stats.norm.pdf(sorted(day_wind_speed), day_mean, day_sigma))
# plt.hist(sorted(day_wind_speed), normed=True)
# plt.title("Day")
# plt.show()

night_df = pd.concat((prior_missing_data_night, after_missing_data_night))
night_wind_speed = night_df['WIND_SPEED'].values
night_mean = np.mean(night_wind_speed)
night_sigma = np.std(night_wind_speed)
night_variance = night_sigma ** 2
# fig = plt.figure()
# plt.plot(sorted(night_wind_speed), stats.norm.pdf(sorted(night_wind_speed), night_mean, night_sigma))
# plt.title("Night")
# plt.show()

print " Day Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" % (day_mean, day_sigma, day_variance)
print " Night Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" % (night_mean, night_sigma, night_variance)
day_tau = 1.0 / day_variance
night_tau = 1.0 / night_variance
hour_f = full_data_jul.index.hour
day_selector_f = ((06 <= hour_f) & (hour_f <= 18))
night_selector_f = ((05 >= hour_f) | (hour_f >= 19))
full_data_jul_day = full_data_jul[day_selector_f]
# full_data_jul_day = full_data_jul_day[(full_data_jul_day['WIND_SPEED'] > 5.0)]
# print min(full_data_jul_day['WIND_SPEED'])daily pattern wind speed
# full_data_jul_day['WIND_SPEED'] = np.where(full_data_jul_day['WIND_SPEED'] == 999, None,  full_data_jul_day['WIND_SPEED'])

# full_data_jul_day.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tub_test.csv')
# raise SystemExit(0)
full_data_jul_day_wind_speed = full_data_jul_day['WIND_SPEED'].values
full_data_jul_night = full_data_jul[night_selector_f]
full_data_jul_night_wind_speed = full_data_jul_night['WIND_SPEED'].values
# fig = plt.figure()
# plt.plot(full_data_jul_day.index, full_data_jul_day['WIND_SPEED'], 'o')
# plt.show()
# raise SystemExit(0)


def day_missing_model():
    # Mean
    mu = day_mean
    # Tau
    tau = day_tau
    # first cutoff
    cutoff_a = 0
    print cutoff_a
    # second cutoff
    cutoff_b = 30
    print cutoff_b
    masked_values = np.ma.masked_values(full_data_jul_day_wind_speed, value=None)
    print masked_values.data.min()
    print masked_values.data.max()
    wind_speed_day = TruncatedNormal('dws', mu, tau, a=cutoff_a, b=cutoff_b, value=masked_values, observed=True)
    return locals()


M_missing_day = pm.MCMC(day_missing_model())
M_missing_day.sample(iter=no_of_iterations, burn=burn, thin=thin)
missing_values = np.mean(np.array(M_missing_day.trace('dws')[-50:-1]), axis=0)
# print len(missing_values)
# print len(full_data_jul_day.index)
masked_values = np.ma.masked_values(full_data_jul_day_wind_speed, value=None)
masked_values[masked_values.mask] = missing_values
full_data_jul_day['corrected_wind_speed'] = np.array(masked_values.data.tolist())
full_data_jul_day['corrected_wind_speed'] = full_data_jul_day['corrected_wind_speed'].astype(int)
# plt.figure()
# plt.plot(full_data_jul_day.index, full_data_jul_day['corrected_wind_speed'], 'ro', label="Missing")
# plt.plot(full_data_jul_day.index, full_data_jul_day['WIND_SPEED'], 'go', label="Observed")
# plt.title("Day Wind Speed Values")
# plt.legend()
# plt.show()


def night_missing_model():
    # Mean
    mu = night_mean
    # Tau
    tau = night_tau
    # first cutoff
    cutoff_a = 0
    # second cutoff
    cutoff_b = 22
    masked_values = np.ma.masked_values(full_data_jul_night_wind_speed, value=None)
    print masked_values.mask.sum()
    wind_speed_day = TruncatedNormal('nws', mu, tau, cutoff_a, cutoff_b, value=masked_values, observed=True)
    return locals()


M_missing_night = pm.MCMC(night_missing_model())
M_missing_night.sample(iter=no_of_iterations, burn=burn, thin=thin)
missing_values = np.mean(np.array(M_missing_night.trace('nws')[-50:-1]), axis=0)
# print len(missing_values)
# print len(full_data_jul_night.index)
masked_values = np.ma.masked_values(full_data_jul_night_wind_speed, value=None)
masked_values[masked_values.mask] = missing_values
full_data_jul_night['corrected_wind_speed'] = np.array(masked_values.data.tolist())
full_data_jul_night['corrected_wind_speed'] = full_data_jul_night['corrected_wind_speed'].astype(int)
# print full_data_jul_night.head()
# plt.figure()
# plt.plot(full_data_jul_night.index, masked_values, 'ro', label="Missing")
# plt.plot(full_data_jul_night.index, full_data_jul_night['WIND_SPEED'], 'go', label="Observed")
# plt.title("Night Wind Speed")
# plt.legend()
# plt.show()
full_data_jul = pd.concat((full_data_jul_day, full_data_jul_night))
full_data_jul.sort_index(inplace=True)
verify_dataframe(full_data_jul)
# fig = plt.figure()
# plt.plot(full_data_jul.index, full_data_jul['corrected_wind_speed'], '-o')
# plt.show()
"""
 Missing data from July 16 to July 17
"""
prior_missing_data = weather_ksndmc_df['2014-07-09':'2014-07-16 12:15:00']
after_missing_data = weather_ksndmc_df['2014-07-17 20:45:00': '2014-07-24']
# check for none values in selected data, if none is present, np.mean will not work
for index in prior_missing_data.index:
    if prior_missing_data['WIND_SPEED'][index] is None:
        print index
for index in after_missing_data.index:
    if after_missing_data['WIND_SPEED'][index] is None:
        print index
full_data_jul_16 = weather_ksndmc_df['2014-07-03':'2014-07-30']
missing_values = np.ma.masked_values(full_data_jul_16['WIND_SPEED'].values, value=None)
print missing_values.mask.sum()
hour_p = prior_missing_data.index.hour
day_selector_p = ((06 <= hour_p) & (hour_p <= 18))
night_selector_p = ((05 >= hour_p) | (hour_p >= 19))
prior_missing_data_day = prior_missing_data[day_selector_p]
prior_missing_data_night = prior_missing_data[night_selector_p]
hour_a = after_missing_data.index.hour
day_selector_a = ((06 <= hour_a) & (hour_a <= 18))
night_selector_a = ((05 >= hour_a) | (hour_a >= 19))
after_missing_data_day = after_missing_data[day_selector_a]
after_missing_data_night = after_missing_data[night_selector_a]
day_df = pd.concat((prior_missing_data_day, after_missing_data_day))
# day_df = day_df[day_df['WIND_SPEED'] > 5.0]
day_wind_speed = day_df['WIND_SPEED'].values
day_mean = np.mean(day_wind_speed)
day_sigma = np.std(day_wind_speed, ddof=1)
day_variance = day_sigma ** 2
# fig = plt.figure()
# plt.plot(sorted(day_wind_speed), stats.norm.pdf(sorted(day_wind_speed), day_mean, day_sigma))
# plt.hist(sorted(day_wind_speed), normed=True)
# plt.title("Day")
# plt.show()

night_df = pd.concat((prior_missing_data_night, after_missing_data_night))
night_wind_speed = night_df['WIND_SPEED'].values
night_mean = np.mean(night_wind_speed)
night_sigma = np.std(night_wind_speed)
night_variance = night_sigma ** 2
# fig = plt.figure()
# plt.plot(sorted(night_wind_speed), stats.norm.pdf(sorted(night_wind_speed), night_mean, night_sigma))
# plt.title("Night")
# plt.show()

print " Day Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" % (day_mean, day_sigma, day_variance)
print " Night Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" % (night_mean, night_sigma, night_variance)
day_tau = 1.0 / day_variance
night_tau = 1.0 / night_variance
hour_f = full_data_jul_16.index.hour
day_selector_f = ((06 <= hour_f) & (hour_f <= 18))
night_selector_f = ((05 >= hour_f) | (hour_f >= 19))
full_data_jul_16_day = full_data_jul_16[day_selector_f]
# full_data_jul_16_day = full_data_jul_16_day[(full_data_jul_16_day['WIND_SPEED'] > 5.0)]
# print min(full_data_jul_16_day['WIND_SPEED'])daily pattern wind speed
# full_data_jul_16_day['WIND_SPEED'] = np.where(full_data_jul_16_day['WIND_SPEED'] == 999, None,  full_data_jul_16_day['WIND_SPEED'])

# full_data_jul_16_day.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tub_test.csv')
# raise SystemExit(0)
full_data_jul_16_day_wind_speed = full_data_jul_16_day['WIND_SPEED'].values
full_data_jul_16_night = full_data_jul_16[night_selector_f]
full_data_jul_16_night_wind_speed = full_data_jul_16_night['WIND_SPEED'].values
# fig = plt.figure()
# plt.plot(full_data_jul_16_day.index, full_data_jul_16_day['WIND_SPEED'], 'o')
# plt.show()
# raise SystemExit(0)


def day_missing_model():
    # Mean
    mu = day_mean
    # Tau
    tau = day_tau
    # first cutoff
    cutoff_a = 0
    print cutoff_a
    # second cutoff
    cutoff_b = 30
    print cutoff_b
    masked_values = np.ma.masked_values(full_data_jul_16_day_wind_speed, value=None)
    print masked_values.data.min()
    print masked_values.data.max()
    wind_speed_day = TruncatedNormal('dws', mu, tau, a=cutoff_a, b=cutoff_b, value=masked_values, observed=True)
    return locals()


M_missing_day = pm.MCMC(day_missing_model())
M_missing_day.sample(iter=no_of_iterations, burn=burn, thin=thin)
missing_values = np.mean(np.array(M_missing_day.trace('dws')[-50:-1]), axis=0)
# print len(missing_values)
# print len(full_data_jul_16_day.index)
masked_values = np.ma.masked_values(full_data_jul_16_day_wind_speed, value=None)
masked_values[masked_values.mask] = missing_values
full_data_jul_16_day['corrected_wind_speed'] = np.array(masked_values.data.tolist())
full_data_jul_16_day['corrected_wind_speed'] = full_data_jul_16_day['corrected_wind_speed'].astype(int)
# plt.figure()
# plt.plot(full_data_jul_16_day.index, full_data_jul_16_day['corrected_wind_speed'], 'ro', label="Missing")
# plt.plot(full_data_jul_16_day.index, full_data_jul_16_day['WIND_SPEED'], 'go', label="Observed")
# plt.title("Day Wind Speed Values")
# plt.legend()
# plt.show()
def night_missing_model():
    # Mean
    mu = night_mean
    # Tau
    tau = night_tau
    # first cutoff
    cutoff_a = 0
    # second cutoff
    cutoff_b = 27
    masked_values = np.ma.masked_values(full_data_jul_16_night_wind_speed, value=None)
    print masked_values.mask.sum()
    wind_speed_day = TruncatedNormal('nws', mu, tau, cutoff_a, cutoff_b, value=masked_values, observed=True)
    return locals()


M_missing_night = pm.MCMC(night_missing_model())
M_missing_night.sample(iter=no_of_iterations, burn=burn, thin=thin)
missing_values = np.mean(np.array(M_missing_night.trace('nws')[-50:-1]), axis=0)
# print len(missing_values)
# print len(full_data_jul_16_night.index)
masked_values = np.ma.masked_values(full_data_jul_16_night_wind_speed, value=None)
masked_values[masked_values.mask] = missing_values
full_data_jul_16_night['corrected_wind_speed'] = np.array(masked_values.data.tolist())
full_data_jul_16_night['corrected_wind_speed'] = full_data_jul_16_night['corrected_wind_speed'].astype(int)
# # print full_data_jul_16_night.head()
# # plt.figure()
# # plt.plot(full_data_jul_16_night.index, masked_values, 'ro', label="Missing")
# # plt.plot(full_data_jul_16_night.index, full_data_jul_16_night['WIND_SPEED'], 'go', label="Observed")
# # plt.title("Night Wind Speed")
# # plt.legend()
# # plt.show()
full_data_jul_16 = pd.concat((full_data_jul_16_day, full_data_jul_16_night))
full_data_jul_16.sort_index(inplace=True)
verify_dataframe(full_data_jul_16)
# fig = plt.figure()
# plt.plot(full_data_jul_16.index, full_data_jul_16['corrected_wind_speed'], '-o')
# plt.show()
"""
 Missing data from July 25 to July 26
"""
prior_missing_data = weather_ksndmc_df['2014-07-21':'2014-07-25 13:00:00']
after_missing_data = weather_ksndmc_df['2014-07-26 20:45:00': '2014-07-30']
# check for none values in selected data, if none is present, np.mean will not work
for index in prior_missing_data.index:
    if prior_missing_data['WIND_SPEED'][index] is None:
        print index
for index in after_missing_data.index:
    if after_missing_data['WIND_SPEED'][index] is None:
        print index
full_data_jul_25 = weather_ksndmc_df['2014-07-21':'2014-07-30']
missing_values = np.ma.masked_values(full_data_jul_25['WIND_SPEED'].values, value=None)
# print missing_values.mask.sum()
hour_p = prior_missing_data.index.hour
day_selector_p = ((06 <= hour_p) & (hour_p <= 18))
night_selector_p = ((05 >= hour_p) | (hour_p >= 19))
prior_missing_data_day = prior_missing_data[day_selector_p]
prior_missing_data_night = prior_missing_data[night_selector_p]
hour_a = after_missing_data.index.hour
day_selector_a = ((06 <= hour_a) & (hour_a <= 18))
night_selector_a = ((05 >= hour_a) | (hour_a >= 19))
after_missing_data_day = after_missing_data[day_selector_a]
after_missing_data_night = after_missing_data[night_selector_a]
day_df = pd.concat((prior_missing_data_day, after_missing_data_day))
# day_df = day_df[day_df['WIND_SPEED'] > 5.0]
day_wind_speed = day_df['WIND_SPEED'].values
day_mean = np.mean(day_wind_speed)
day_sigma = np.std(day_wind_speed, ddof=1)
day_variance = day_sigma ** 2
# fig = plt.figure()
# plt.plot(sorted(day_wind_speed), stats.norm.pdf(sorted(day_wind_speed), day_mean, day_sigma))
# plt.hist(sorted(day_wind_speed), normed=True)
# plt.title("Day")
# plt.show()

night_df = pd.concat((prior_missing_data_night, after_missing_data_night))
night_wind_speed = night_df['WIND_SPEED'].values
night_mean = np.mean(night_wind_speed)
night_sigma = np.std(night_wind_speed)
night_variance = night_sigma ** 2
# fig = plt.figure()
# plt.plot(sorted(night_wind_speed), stats.norm.pdf(sorted(night_wind_speed), night_mean, night_sigma))
# plt.title("Night")
# plt.show()

print " Day Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" % (day_mean, day_sigma, day_variance)
print " Night Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" % (night_mean, night_sigma, night_variance)
day_tau = 1.0 / day_variance
night_tau = 1.0 / night_variance
hour_f = full_data_jul_25.index.hour
day_selector_f = ((06 <= hour_f) & (hour_f <= 18))
night_selector_f = ((05 >= hour_f) | (hour_f >= 19))
full_data_jul_25_day = full_data_jul_25[day_selector_f]
# full_data_jul_25_day = full_data_jul_25_day[(full_data_jul_25_day['WIND_SPEED'] > 5.0)]
# print min(full_data_jul_25_day['WIND_SPEED'])daily pattern wind speed
# full_data_jul_25_day['WIND_SPEED'] = np.where(full_data_jul_25_day['WIND_SPEED'] == 999, None,  full_data_jul_25_day['WIND_SPEED'])

# full_data_jul_25_day.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tub_test.csv')
# raise SystemExit(0)
full_data_jul_25_day_wind_speed = full_data_jul_25_day['WIND_SPEED'].values
full_data_jul_25_night = full_data_jul_25[night_selector_f]
full_data_jul_25_night_wind_speed = full_data_jul_25_night['WIND_SPEED'].values
# fig = plt.figure()
# plt.plot(full_data_jul_25_day.index, full_data_jul_25_day['WIND_SPEED'], 'o')
# plt.show()
# raise SystemExit(0)

def day_missing_model():
    # Mean
    mu = day_mean
    # Tau
    tau = day_tau
    # first cutoff
    cutoff_a = 0
    print cutoff_a
    # second cutoff
    cutoff_b = 30
    print cutoff_b
    masked_values = np.ma.masked_values(full_data_jul_25_day_wind_speed, value=None)
    print masked_values.data.min()
    print masked_values.data.max()
    wind_speed_day = TruncatedNormal('dws', mu, tau, a=cutoff_a, b=cutoff_b, value=masked_values, observed=True)
    return locals()


M_missing_day = pm.MCMC(day_missing_model())
M_missing_day.sample(iter=no_of_iterations, burn=burn, thin=thin)
missing_values = np.mean(np.array(M_missing_day.trace('dws')[-50:-1]), axis=0)
# print len(missing_values)
# print len(full_data_jul_25_day.index)
masked_values = np.ma.masked_values(full_data_jul_25_day_wind_speed, value=None)
masked_values[masked_values.mask] = missing_values
full_data_jul_25_day['corrected_wind_speed'] = np.array(masked_values.data.tolist())
full_data_jul_25_day['corrected_wind_speed'] = full_data_jul_25_day['corrected_wind_speed'].astype(int)
# plt.figure()
# plt.plot(full_data_jul_25_day.index, full_data_jul_25_day['corrected_wind_speed'], 'ro', label="Missing")
# plt.plot(full_data_jul_25_day.index, full_data_jul_25_day['WIND_SPEED'], 'go', label="Observed")
# plt.title("Day Wind Speed Values")
# plt.legend()
# plt.show()


def night_missing_model():
    # Mean
    mu = night_mean
    # Tau
    tau = night_tau
    # first cutoff
    cutoff_a = 0
    # second cutoff
    cutoff_b = 27
    masked_values = np.ma.masked_values(full_data_jul_25_night_wind_speed, value=None)
    print masked_values.mask.sum()
    wind_speed_day = TruncatedNormal('nws', mu, tau, cutoff_a, cutoff_b, value=masked_values, observed=True)
    return locals()


M_missing_night = pm.MCMC(night_missing_model())
M_missing_night.sample(iter=no_of_iterations, burn=burn, thin=thin)
missing_values = np.mean(np.array(M_missing_night.trace('nws')[-50:-1]), axis=0)
# print len(missing_values)
# print len(full_data_jul_25_night.index)
masked_values = np.ma.masked_values(full_data_jul_25_night_wind_speed, value=None)
masked_values[masked_values.mask] = missing_values
full_data_jul_25_night['corrected_wind_speed'] = np.array(masked_values.data.tolist())
full_data_jul_25_night['corrected_wind_speed'] = full_data_jul_25_night['corrected_wind_speed'].astype(int)
# print full_data_jul_25_night.head()
# plt.figure()
# plt.plot(full_data_jul_25_night.index, masked_values, 'ro', label="Missing")
# plt.plot(full_data_jul_25_night.index, full_data_jul_25_night['WIND_SPEED'], 'go', label="Observed")
# plt.title("Night Wind Speed")
# plt.legend()
# plt.show()
full_data_jul_25 = pd.concat((full_data_jul_25_day, full_data_jul_25_night))
full_data_jul_25.sort_index(inplace=True)
verify_dataframe(full_data_jul_25)
complete_missing_data = pd.concat((full_data, full_data_jan, full_data_jul, full_data_jul_16, full_data_jul_16))
print len(complete_missing_data.index)
# raise SystemExit(0)
complete_missing_data.to_csv("/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tubgere_weather_missing_wind_speed.csv")
date_format = '%Y-%m-%d %H:%M:%S'
# weather_ksndmc_df = weather_ksndmc_df[:'2015-01-27']
print "i"
weather_ksndmc_df['WIND_SPEED'].ix[complete_missing_data.index] = complete_missing_data['corrected_wind_speed']
weather_ksndmc_df['WIND_SPEED'] = weather_ksndmc_df["WIND_SPEED"]/10.0
weather_ksndmc_df.index.name = "Date_Time"
print weather_ksndmc_df['2015-01-27']
weather_ksndmc_df.to_csv("/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tubgere_weather_corrected_wind_speed.csv")
# fig = plt.figure()
# plt.plot(full_data_jul_25.index, full_data_jul_25['corrected_wind_speed'], '-o')
# plt.show()
