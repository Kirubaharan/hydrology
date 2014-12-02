__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
# import seaborn as sns
from matplotlib import rc
import pickle
from datetime import timedelta
import brewer2mpl
from matplotlib import cm
from scipy.optimize import curve_fit
import matplotlib as mpl



# latex parameters
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=24)

# style
# mpl.rcParams.update(mpl.rcParamsDefault)
daily_format = '%Y-%m-%d'
datetime_format = '%Y-%m-%d %H:%M:%S'
# colorbrewer
dark2_colors = brewer2mpl.get_map('Dark2', 'Qualitative', 7).mpl_colors

file_591 = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/et_infilt_591.csv'
file_599 = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_599/daily_wb_599.CSV'
rain_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/corrected_rain.csv'
rain_df = pd.read_csv(rain_file, sep=',', header=0)
rain_df['Date_Time'] = pd.to_datetime(rain_df['Date_Time'], format=datetime_format)
rain_df.set_index(rain_df['Date_Time'], inplace=True)
# sort based on index
rain_df.sort_index(inplace=True)
# drop date time column
rain_df = rain_df.drop('Date_Time', 1)
rain_df = rain_df.resample('D', how=np.sum)
wb_591 = pd.read_csv(file_591, sep=',', header=0)
wb_599 = pd.read_csv(file_599, sep=',', header=0)
wb_591.set_index(pd.to_datetime(wb_591['Date'], format=daily_format), inplace=True)
wb_599.set_index(pd.to_datetime(wb_599['Date'], format=daily_format), inplace=True)
del wb_591['Date']
del wb_599['Date']
# missing date time for 599
with open("/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_599/initial_time.pickle", "rb") as f:
    initial_time = pickle.load(f)
with open("/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_599/final_time.pickle", "rb") as f:
    final_time = pickle.load(f)

initial_time_591 = initial_time.strftime(daily_format)
final_time_591 =  final_time.strftime(daily_format)
missing_data_days = (final_time-initial_time).days
print missing_data_days
wb_599 = wb_599[wb_599['infiltration(cu.m)'] > 0]


def annotate_dim(ax,xyfrom,xyto,text=None):
    ax.annotate("",xyfrom,xyto,arrowprops=dict(arrowstyle='<->', linewidth=1.5))
    # ax.annotate(text, xy=(xyfrom[0]+ timedelta(days=17), xyfrom[1]),xycoords='data', xytext=(-10,-10), textcoords='offset points')

fig, (ax1, ax2) = plt.subplots(nrows=2,ncols=1, sharex=True, figsize=(16, 8))
bar_1_1 = ax1.bar(rain_df.index, rain_df['Rain Collection (mm)'], 0.45, color='b',alpha=0.5, label = 'Rainfall (mm)', rasterized=True)
ax1.invert_yaxis()
for t1 in ax1.get_yticklabels():
    t1.set_color('b')
ax1_1 = ax1.twinx()
bar_2_1 = ax2.bar(rain_df.index, rain_df['Rain Collection (mm)'], 0.45, color='b',alpha=0.5, label = 'Rainfall (mm)',rasterized=True)
ax2.invert_yaxis()
for t1 in ax2.get_yticklabels():
    t1.set_color('b')
ax2_1 = ax2.twinx()
bar_1 = ax1_1.bar(wb_591.index, wb_591['infiltration(cu.m)'], 0.45, color='g',alpha=0.85,label=r"\textbf{Infiltration ($m^3/day$}",rasterized=True)
bar_2 = ax2_1.bar(wb_599.index, wb_599['infiltration(cu.m)'], 0.45, color='g',alpha=0.85,rasterized=True)
bar_1_2 = ax1_1.bar(wb_591.index, wb_591['Evaporation (cu.m)'], 0.45, color='r',alpha=0.85, label=r"\textbf{Evaporation ($m^3/day$)}",rasterized=True)
bar_2_2 = ax2_1.bar(wb_599.index, wb_599['Evaporation (cu.m)'], 0.45, color='r',alpha=0.85, label=r"\textbf{Evaporation ($m^3/day$)}",rasterized=True)
bracket = annotate_dim(ax2_1, xyfrom=[initial_time_591,1], xyto=[final_time_591,1], text='Missing Data')
text = ax2_1.text(initial_time+timedelta(days=10), 2, "Missing Data",rasterized=True)
lns = [bar_1_1, bar_1, bar_1_2]
labs = [r'\textbf{Rainfall ($mm$)}', r"\textbf{Infiltration ($m^3/day$)}", r"\textbf{Evaporation ($m^3/day$)}",]
ax2_1.legend(lns, labs, loc='upper center', fancybox=True, ncol=3, bbox_to_anchor=(0.5, -0.05),prop={'size':16})
fig.text(0.06, 0.5, 'Rainfall (mm)', ha='center', va='center', rotation='vertical')
plt.figtext(0.95, 0.5, r'Evaporation/Infiltration ($m^3/day)$', ha='center', va='center', rotation='vertical')
ax1.set_title("Check Dam 591", fontsize=26)
ax2_1.set_title("Check Dam 599", fontsize=26)
# increase tick label size
# left y axis 1
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
# left y axis 2
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
# xaxis
for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax1_1.get_yticklabels():
    tick.set_fontsize(14)
for tick in ax2_1.get_yticklabels():
    tick.set_fontsize(14)
plt.savefig('/media/kiruba/New Volume/AGU/poster/agu_checkdam/image/evap_infilt_agu.pdf', dpi=400)
# plt.show()
# pie charts
# 591
evap_591 = wb_591['Evaporation (cu.m)'].sum()
infilt_591 = wb_591['infiltration(cu.m)'].sum()
evap_599 = wb_599['Evaporation (cu.m)'].sum()
infilt_599 = wb_599['infiltration(cu.m)'].sum()
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(16, 8), facecolor='white' )
pie_1, texts_1, autotexts_1 = ax1.pie([evap_591,infilt_591], labels=['E', 'I'],
                colors=['#e34a33', '#2ca25f'],
                autopct='%i%%',
                explode=(0,0.15),
                startangle=90)
pie_2, texts_2, autotexts_2 = ax2.pie([evap_599,infilt_599],
                labels=['E', 'I'],
                colors=['#e34a33', '#2ca25f'],
                autopct='%i%%',
                explode=(0,0.15),
                startangle=90)
ax1.axis('equal')
ax2.axis('equal')
plt.tight_layout()
ax1.set_title('Check Dam 591', fontsize=26)
ax2.set_title('Check Dam 599',fontsize=26)
ax1.set_axis_bgcolor('white')
ax2.set_axis_bgcolor('white')

ax2.legend(["Evaporation", "Infiltration"],fancybox=True,prop={'size':16})
for label in texts_1:
    label.set_fontsize(20)
for label in texts_2:
    label.set_fontsize(20)
plt.savefig('/media/kiruba/New Volume/AGU/poster/agu_checkdam/image/pie_evap_infilt.pdf', dpi=400)
# plt.show()
# print wb_591.head()
dry_water_balance_591 = wb_591[wb_591['status'] == 'N']
dry_water_balance_599 = wb_599[wb_599['status'] == 'N']
stage_cal_591 = dry_water_balance_591['stage(m)']
stage_cal_599 = dry_water_balance_599['stage(m)']
inf_cal_591 = dry_water_balance_591['infiltration rate (m/day)']
inf_cal_599 = dry_water_balance_599['infiltration rate (m/day)']


def func(h, alpha, beta):
    return alpha*(h**beta)

popt_591, pcov_591 = curve_fit(func, stage_cal_591, inf_cal_591)
popt_599, pcov_599 = curve_fit(func, stage_cal_599, inf_cal_599)
stage_cal_new_591 = np.linspace(min(stage_cal_591), max(stage_cal_591), 50)
inf_cal_new_591 = func(stage_cal_new_591, *popt_591)
stage_cal_new_599 = np.linspace(min(stage_cal_599), max(stage_cal_599), 50)
inf_cal_new_599 = func(stage_cal_new_599, *popt_599)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(16, 8), facecolor='white' )
ax1.plot(stage_cal_591, inf_cal_591, 'bo',rasterized=True)
ax1.plot(stage_cal_new_591, inf_cal_new_591, 'r-',rasterized=True)
ax1.text(x=0.4, y=.035, fontsize=15, s=r'Infiltration rate = ${0:.2f}{{h}}^{{{1:.2f}}}$'.format(popt_591[0], popt_591[1]))
ax2.plot(stage_cal_599, inf_cal_599, 'bo',rasterized=True)
ax2.plot(stage_cal_new_599, inf_cal_new_599, 'r-',rasterized=True)
ax2.text(x=0.3, y=.035, fontsize=15, s=r'Infiltration rate = ${0:.2f}{{h}}^{{{1:.2f}}}$'.format(popt_599[0], popt_599[1]))
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_position(('outward', 20))
ax1.spines['left'].set_position(('outward', 30))
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
ax1.grid(False)
ax1.set_axis_bgcolor('white')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_position(('outward', 20))
ax2.spines['left'].set_position(('outward', 30))
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')
ax2.grid(False)
ax2.set_axis_bgcolor('white')
fig.text(0.5, 0.02, 'Stage (m)', ha='center', va='center')
plt.figtext(0.03, 0.5, r'Infiltration rate ($m/day)$', ha='center', va='center', rotation='vertical')
ax1.set_title("Check Dam 591", fontsize=26)
ax2.set_title("Check Dam 599", fontsize=26)
ax2.set_ylim(ax1.get_ylim())
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax2.legend(["Observation", "Prediction"],fancybox=True,prop={'size':16})
plt.savefig('/media/kiruba/New Volume/AGU/poster/agu_checkdam/image/stage_infilt.pdf', dpi=400)
plt.show()