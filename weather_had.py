__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import checkdam.checkdam as cd
from datetime import timedelta
import datetime
import pymc as pm

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)

# aral_rain_file_1 = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/kanaswadi/KSNDMC_01-05-2014_10-09-2014_KANASAWADI.csv'
aral_rain_file_1 = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tubgere_1_09_14_10_02_15.csv'

aral_rain_df_1 = pd.read_csv(aral_rain_file_1, sep=',', header=0)
aral_rain_df_1.drop(['Sl no', 'TRGCODE', 'HOBLINAME'], inplace=True, axis=1)
# print aral_rain_df_1.head()
# raise SystemExit(0)
# aral_rain_df_2  = pd.read_csv(aral_rain_file_2, sep=',')
# aral_rain_df_2.drop(['Sl no', 'TRGCODE', 'HOBLINAME'], inplace=True, axis=1)
# print aral_rain_df_2.head()
data_1 = []
for row_no, row in aral_rain_df_1.iterrows():
    date = row['Date']
    for time, value in row.ix[1:, ].iteritems():
        data_1.append((date, time, value))

data_1_df = pd.DataFrame(data_1,columns=['date', 'time', 'rain(mm)'])


date_format_1 = "%d-%b-%y %H:%M"
data_1_df['date_time'] = pd.to_datetime(data_1_df['date'] + ' ' + data_1_df['time'], format=date_format_1)
data_1_df.set_index(data_1_df['date_time'], inplace=True)
data_1_df.sort_index(inplace=True)
data_1_df.drop(['date_time', 'date', 'time'], axis=1, inplace=True)


# cumulative difference
data_1_8h_df = data_1_df['2014-09-01 8H30T': '2015-02-09 8H30T']
data_1_8h_df['diff'] = 0.000

for d1, d2 in cd.pairwise(data_1_8h_df.index):
    if data_1_8h_df['rain(mm)'][d2] > data_1_8h_df['rain(mm)'][d1]:
        data_1_8h_df['diff'][d2] = data_1_8h_df['rain(mm)'][d2] - data_1_8h_df['rain(mm)'][d1]
        
data_1_30min_df = data_1_8h_df.resample('30Min', how=np.sum, label='right', closed='right')
aral_rain_df = data_1_30min_df

"""
Remove duplicates
"""
aral_rain_df['index'] = aral_rain_df.index
aral_rain_df.drop_duplicates(subset='index', take_last=True, inplace=True)
del aral_rain_df['index']
aral_rain_df = aral_rain_df.sort()
# print aral_rain_df.head()

"""
Weather
"""
weather_ksndmc = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tubgere_weather_01May14_10Feb15.csv'
weather_ksndmc_df = pd.read_csv(weather_ksndmc, sep=',')
weather_ksndmc_df.drop(['Sl no', 'HOBLI'], inplace=True, axis=1)
weather_date_format = "%d-%b-%y %H:%M:%S+05:30"
weather_ksndmc_df['date_time'] = pd.to_datetime(weather_ksndmc_df['DATE'] + " " + weather_ksndmc_df['TIME'], format=weather_date_format)
weather_ksndmc_df.set_index(weather_ksndmc_df['date_time'], inplace=True)
weather_ksndmc_df.sort_index(inplace=True)
cols = weather_ksndmc_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
weather_ksndmc_df = weather_ksndmc_df[cols]
weather_ksndmc_df.drop(['date_time', 'DATE', 'TIME'], inplace=True, axis=1)

# weather_ksndmc_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/KSNDMC/Tubgere_weather_01May14_10Feb15_corrected.csv')

# fig = plt.figure()
# plt.plot(weather_regr_df['HUMIDITY'], weather_regr_df['TEMPERATURE'], 'go')
# plt.show()

minute = weather_ksndmc_df.index.minute
weather_ksndmc_df = weather_ksndmc_df[((minute == 0) | (minute == 15) | (minute == 30) | (minute == 45) | (minute == 60))]
weather_ksndmc_df['index'] = weather_ksndmc_df.index
weather_ksndmc_df.drop_duplicates(subset='index', take_last=True, inplace=True)
del weather_ksndmc_df['index']
weather_ksndmc_df = weather_ksndmc_df.sort()
weather_ksndmc_df['WIND_SPEED'] = np.where(weather_ksndmc_df['WIND_SPEED'] > 3.0, np.nan, weather_ksndmc_df['WIND_SPEED'])
start_time = min(weather_ksndmc_df.index)
end_time = max(weather_ksndmc_df.index)
new_index = pd.date_range(start=start_time, end=end_time, freq='15min')
weather_ksndmc_df = weather_ksndmc_df.reindex(new_index)
print len(new_index)
Z = weather_ksndmc_df['WIND_SPEED'].values
print Z

switch = pm.DiscreteUniform('switch', lower=0, upper=27406)
early_mean = pm.Exponential('early_mean', beta=1)
late_mean = pm.Exponential('late_mean', beta=1)


def rate(s, e, l):
    """Allocate appropriate mean to time series"""
    out = np.empty(len(Z))
    # Early mean prior to switchpoint
    out[:s] = e
    # Late mean following switchpoint
    out[s:] = l
    return out
rate = rate(s=switch, e=early_mean, l=late_mean)
print rate
masked_values = np.ma.masked_equal(Z, value=np.nan)
wind_speed = pm.Poisson('wind_speed', mu=rate, value=masked_values, observed=True)

# print Z[:5,1:4]
# print Z.dtype
#
# # print weather_ksndmc_df.head()
# X = Z[ :,2:4]
# # print X
# ## Missing data patterns
# ioo = np.flatnonzero(np.isfinite(X).all(1))
# iom = np.flatnonzero(np.isfinite(X[:,0]) & np.isnan(X[:,1]))
# imo = np.flatnonzero(np.isnan(X[:,0]) & np.isfinite(X[:,1]))
# imm = np.flatnonzero(np.isnan(X).all(1))
# ## Complete data
# XC = X[ioo,:]
# ## Number of multiple imputation iterations
# nmi = 100
# ## Do the multiple imputation
# F = np.zeros(nmi, dtype=np.float64)
# for j in range(nmi):
#     ## Bootstrap the complete data
#     ii = np.random.randint(0, len(ioo), len(ioo))
#     XB = XC[ii,:]
#     ## Column-wise means
#     X_mean = XB.mean(0)
#     ## Column-wise standard deviations
#     X_sd = XB.std(0)
#     ## Correlation coefficient
#     r = np.corrcoef(XB.T)[0,1]
#     ## The imputed data
#     XI = X.copy()
#     ## Impute the completely missing rows
#     Q = np.random.normal(size=(X.shape[0],2))
#     Q[:,1] = r*Q[:,0] + np.sqrt(1 - r**2)*Q[:,1]
#     Q = Q*X_sd + X_mean
#     XI[imm,:] = Q[imm,:]
#
#     ## Impute the rows with missing first column
#     ## using the conditional distribution
#     va = X_sd[0]**2 - r**2/X_sd[1]**2
#     XI[imo,0] = r*X[imo,1]*(X_sd[0]/X_sd[1]) +\
#                 np.sqrt(va)*np.random.normal(size=len(imo))
#
#     ## Impute the rows with missing second column
#     ## using the conditional distribution
#     va = X_sd[1]**2 - r**2/X_sd[0]**2
#     XI[iom,1] = r*X[iom,0]*(X_sd[1]/X_sd[0]) +\
#                 np.sqrt(va)*np.random.normal(size=len(iom))
#
#     ## The correlation coefficient of the imputed data
#     r = np.corrcoef(XI[:,0], XI[:,1])[0,1]
#
#     ## The Fisher-transformed correlation coefficient
#     F[j] = 0.5*np.log((1+r) / (1-r))
#
# ## Apply the combining rule, see, e.g.
# ## http://sites.stat.psu.edu/~jls/mifaq.html#howto
# FM = F.mean()
# RM = (np.exp(2*FM)-1) / (np.exp(2*FM)+1)
# VA = (1 + 1/float(nmi))*F.var() + 1/float(Z.shape[0]-3)
# SE = np.sqrt(VA)
# LCL,UCL = FM-2*SE,FM+2*SE
# LCL = (np.exp(2*LCL)-1) / (np.exp(2*LCL)+1)
# UCL = (np.exp(2*UCL)-1) / (np.exp(2*UCL)+1)
#
# print "\nMultiple imputation:"
# print "%.2f(%.2f,%.2f)" % (RM, LCL, UCL)
# print XI
# wind_speed =
fig = plt.figure()
plt.plot_date(weather_ksndmc_df.index, wind_speed, 'r-o')
plt.show()

raise SystemExit(0)

start_time = min(weather_ksndmc_df.index)
end_time = max(weather_ksndmc_df.index)
new_index = pd.date_range(start=start_time, end=end_time, freq='15min')
weather_ksndmc_df = weather_ksndmc_df.reindex(new_index, method=None)
# weather_ksndmc_df = weather_ksndmc_df.interpolate(method='time')
# print weather_ksndmc_df['2014-06-04 07:00:00':]
weather_ksndmc_df = weather_ksndmc_df.resample('30Min', how=np.mean, label='right', closed='right')
weather_regr_df = weather_ksndmc_df[:'2014-05-17']
# weather_ksndmc_df = weather_ksndmc_df[ :" 2015-02-09"]
# print weather_ksndmc_df.tail()

base_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/had_june_jan_10.csv'
may_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/Hadonahalli_WS_May 2014.csv'
#read csv file
df_base = pd.read_csv(base_file, header=0, sep=',')
may_df = pd.read_csv(may_file, header=0, sep=',')
# may_df['Time'] = may_df['Time'].map(lambda x: x[ :5])
# print may_df.head()
#Drop seconds
df_base['Time'] = df_base['Time'].map(lambda x: x[ :5])
# convert date and time columns into timestamp
# print df_base.head()
date_format = '%d/%m/%y %H:%M'
df_base['Date_Time'] = pd.to_datetime(df_base['Date'] + ' ' + df_base['Time'], format=date_format)
df_base.set_index(df_base['Date_Time'], inplace=True)
date_format = '%d/%m/%y %H:%M:%S'
# print may_df['Date'][0] + ' ' + may_df['Time'][0]
may_df['Date_Time'] = pd.to_datetime(may_df['Date'] + ' ' +  may_df["Time"], format=date_format)
may_df.set_index(may_df['Date_Time'], inplace=True)
rounded = np.array(may_df.index, dtype='datetime64[m]')
may_df.set_index(rounded,inplace=True)
may_df.columns.values[6] = 'Air Temperature (C)'
may_df.columns.values[7] = 'Min Air Temperature (C)'
may_df.columns.values[8] = 'Max Air Temperature (C)'
may_df.columns.values[15] = 'Canopy Temperature (C)'
may_df['index'] = may_df.index
may_df.drop_duplicates(subset='index', take_last=True, inplace=True)
del may_df['index']
may_df = may_df.sort()
new_index = pd.date_range(start=min(may_df.index), end=max(may_df.index), freq='30min' )
may_df = may_df.reindex(index=new_index, method=None)
may_df = may_df.interpolate(method='time')
# print df_base.columns.values[15]
df_base.columns.values[6] = 'Air Temperature (C)'
df_base.columns.values[7] = 'Min Air Temperature (C)'
df_base.columns.values[8] = 'Max Air Temperature (C)'
df_base.columns.values[15] = 'Canopy Temperature (C)'
df_base['index'] = df_base.index
df_base.drop_duplicates(subset='index', take_last=True, inplace=True)
del df_base['index']
df_base = df_base.sort()
new_index = pd.date_range(start=min(df_base.index), end=max(df_base.index), freq='30min' )
df_base = df_base.reindex(index=new_index, method=None)
df_base = df_base.interpolate(method='time')
df_base = pd.concat([may_df, df_base], axis=0)

# ksndmc_cutoff = {'WIND_SPEED':[3,'>']}
# n = 20
# while n > 0:
#     print n
#     w_timestamps_ksndmc = cd.pick_incorrect_value(weather_ksndmc_df, **ksndmc_cutoff)
#     weather_ksndmc_df = cd.day_interpolate(weather_ksndmc_df, 'WIND_SPEED', w_timestamps_ksndmc)
#     n -= 1


fig = plt.figure()
plt.plot_date(weather_ksndmc_df.index, weather_ksndmc_df['WIND_SPEED'], 'ro')
plt.show()

# fig = plt.figure()
# plt.plot_date(df_base.index, df_base['Air Temperature (C)'], 'go')
# plt.show()
# print df_base.tail()
# raise SystemExit(0)

rain_df = aral_rain_df[['diff']]
# rain_df.columns.values[0] = "Rain Collection (mm)"
rain_df_1 = df_base['Rain Collection (mm)'][ : '2014-09-01 8H00T']
rain_df_1 = pd.DataFrame([rain_df_1[i] for i in range(len(rain_df_1))], columns=['diff'], index=rain_df_1.index)
rain_df = pd.concat([rain_df_1, rain_df], axis=0)
rain_df.columns.values[0] = "rain (mm)"


# weather_df.index.name = "Date_Time"
rain_df.index.name = "Date_Time"
# weather_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/ksndmc_weather.csv')
rain_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weather_station/hadonahalli/ksndmc_rain.csv')
# print aral_rain_df.tail()
# print df_base.tail()
# delta = df_base.index[0] - df_base.index[-1]
# print delta.days
# fig= plt.figure()
# plt.plot_date(df_base.index, df_base['Air Temperature (C)'])
# plt.bar(aral_rain_df.index, aral_rain_df['diff'], width=0.02, color='b')
# # plt.bar(df_base.index, df_base['Rain Collection (mm)'],width=0.02, color='g')
# fig.autofmt_xdate(rotation=90)
# plt.show()
# print aral_rain_df['diff'].sum()
# print(df_base['Rain Collection (mm)'].sum())
