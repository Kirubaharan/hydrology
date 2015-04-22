__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import checkdam.checkdam as cd
from datetime import timedelta
import scipy.stats as stats
import matplotlib

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
minute = weather_ksndmc_df.index.minute
weather_ksndmc_df = weather_ksndmc_df[
    ((minute == 0) | (minute == 15) | (minute == 30) | (minute == 45) | (minute == 60))]
# drop duplicates
weather_ksndmc_df['index'] = weather_ksndmc_df.index
weather_ksndmc_df.drop_duplicates(subset='index', take_last=True, inplace=True)
del weather_ksndmc_df['index']
weather_ksndmc_df = weather_ksndmc_df.sort()
start_time = min(weather_ksndmc_df.index)
end_time = max(weather_ksndmc_df.index)
new_index = pd.date_range(start=start_time, end=end_time, freq='15min')
weather_ksndmc_df = weather_ksndmc_df.reindex(new_index, fill_value=5)
weather_ksndmc_df["WIND_SPEED"] = np.where(weather_ksndmc_df["WIND_SPEED"] > 3.0, None,
                                           weather_ksndmc_df['WIND_SPEED'])
max_limit = max(weather_ksndmc_df.index) - timedelta(days=1)
min_limit = min(weather_ksndmc_df.index) + timedelta(days=1)
for index in weather_ksndmc_df.index:
    if weather_ksndmc_df['WIND_SPEED'][index] is None and (index > min_limit) and (index < max_limit):
        previous_day_value = weather_ksndmc_df['WIND_SPEED'][index - timedelta(days=1)]
        next_day_value = weather_ksndmc_df['WIND_SPEED'][index + timedelta(days=1)]
        if (previous_day_value != None) and (next_day_value != None):
            weather_ksndmc_df["WIND_SPEED"][index] = 0.5 * (previous_day_value + next_day_value)

"""
First period missing data
"""
prior_missing_data = weather_ksndmc_df['2014-08-25':'2014-08-28 17:15:00']
after_missing_data = weather_ksndmc_df['2014-09-03 14:30:00': '2014-09-06']
full_data = weather_ksndmc_df['2014-08-25':'2014-09-06']
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
day_wind_speed = day_df['WIND_SPEED'].values
day_mean = np.mean(day_wind_speed)
day_sigma = np.std(day_wind_speed)
day_variance = day_sigma ** 2
night_df = pd.concat((prior_missing_data_night, after_missing_data_night))
night_wind_speed = night_df['WIND_SPEED'].values
night_mean = np.mean(night_wind_speed)
night_sigma = np.std(night_wind_speed)
night_variance = night_sigma ** 2
# Plot
print("Day Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f \n" % (day_mean, day_sigma, day_variance) + "Night Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" % (night_mean, night_sigma, night_variance))
fig = plt.figure()
plt.plot(sorted(day_wind_speed), stats.norm.pdf(sorted(day_wind_speed), day_mean, day_sigma), 'g-o', label='Day')
night_fit = stats.norm.pdf(sorted(night_wind_speed), night_mean, night_sigma)
plt.plot(sorted(night_wind_speed), night_fit, '-ro', label="Night")
# plt.text(0.25, 1.1, "Day Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f \n" % (day_mean, day_sigma, day_variance) + "Night Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" % (night_mean, night_sigma, night_variance), fontsize=28)
plt.title("PDF of wind speed for time period 2014-08-25 : 2014-09-06")
plt.legend().draggable()
plt.show()
raise SystemExit(0)
"""
Missing period from Jan 06 to Jan 12
"""
prior_missing_data = weather_ksndmc_df['2014-12-25':'2015-01-06 21:45:00']
after_missing_data = weather_ksndmc_df['2015-01-12 13:00:00': '2015-01-27']
full_data_jan = weather_ksndmc_df['2014-12-25':'2015-01-27']
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
night_df = pd.concat((prior_missing_data_night, after_missing_data_night))
night_wind_speed = night_df['WIND_SPEED'].values
night_mean = np.mean(night_wind_speed)
night_sigma = np.std(night_wind_speed)
night_variance = night_sigma ** 2
# Plot
fig = plt.figure()
plt.plot(sorted(day_wind_speed), stats.norm.pdf(sorted(day_wind_speed), day_mean, day_sigma), 'g-o', label='Day')
night_fit = stats.norm.pdf(sorted(night_wind_speed), night_mean, night_sigma)
plt.plot(sorted(night_wind_speed), night_fit, '-ro', label="Night")
plt.text(1.04, 0.7, "Day Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f \n" % (day_mean, day_sigma, day_variance) + "Night Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" % (night_mean, night_sigma, night_variance), fontsize=11)
plt.title("PDF of wind speed for time period 2014-12-25 : 2015-01-27")
plt.legend().draggable()
plt.show()
"""
Missing data July 31 to Aug 04
"""
prior_missing_data = weather_ksndmc_df['2014-07-27':'2014-07-31 17:00:00']
after_missing_data = weather_ksndmc_df['2014-08-04 13:00:00': '2014-08-08']
full_data_jul = weather_ksndmc_df['2014-07-27':'2014-08-08']
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
night_df = pd.concat((prior_missing_data_night, after_missing_data_night))
night_wind_speed = night_df['WIND_SPEED'].values
night_mean = np.mean(night_wind_speed)
night_sigma = np.std(night_wind_speed)
night_variance = night_sigma ** 2
# Plot
fig = plt.figure()
plt.plot(sorted(day_wind_speed), stats.norm.pdf(sorted(day_wind_speed), day_mean, day_sigma), 'g-o', label='Day')
night_fit = stats.norm.pdf(sorted(night_wind_speed), night_mean, night_sigma)
plt.plot(sorted(night_wind_speed), night_fit, '-ro', label="Night")
plt.text(1.4, 0.9, "Day Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f \n" % (day_mean, day_sigma, day_variance) + "Night Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" % (night_mean, night_sigma, night_variance), fontsize=11)
plt.title("PDF of wind speed for time period 2014-07-27:2014-08-08")
plt.legend().draggable()
plt.show()
"""
 Missing data from July 16 to July 17
"""
prior_missing_data = weather_ksndmc_df['2014-07-09':'2014-07-16 12:15:00']
after_missing_data = weather_ksndmc_df['2014-07-17 20:45:00': '2014-07-24']
full_data_jul_16 = weather_ksndmc_df['2014-07-03':'2014-07-30']
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
night_df = pd.concat((prior_missing_data_night, after_missing_data_night))
night_wind_speed = night_df['WIND_SPEED'].values
night_mean = np.mean(night_wind_speed)
night_sigma = np.std(night_wind_speed)
night_variance = night_sigma ** 2
# Plot
fig = plt.figure()
plt.plot(sorted(day_wind_speed), stats.norm.pdf(sorted(day_wind_speed), day_mean, day_sigma), 'g-o', label='Day')
night_fit = stats.norm.pdf(sorted(night_wind_speed), night_mean, night_sigma)
plt.plot(sorted(night_wind_speed), night_fit, '-ro', label="Night")
plt.text(1.4, 0.9, "Day Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f \n" % (day_mean, day_sigma, day_variance) + "Night Mean = %0.2f, Sigma = %0.2f, and Variance = %0.2f" % (night_mean, night_sigma, night_variance), fontsize=11)
plt.title("PDF of wind speed for time period 2014-07-03 : 2014-07-30")
plt.legend().draggable()
plt.show()