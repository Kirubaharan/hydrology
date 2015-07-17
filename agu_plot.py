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
from math import sqrt
SPINE_COLOR = 'gray'
import matplotlib
from matplotlib.ticker import MaxNLocator
from statsmodels.nonparametric.smoothers_lowess import lowess


# latex parameters
rc('font', **{'family': 'sans-serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=36)


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES =32
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['\usepackage{gensymb}'],
              'axes.labelsize': 28, # fontsize for x and y labels (was 10)
              'axes.titlesize': 30,
              'text.fontsize': 30, # was 10
              'legend.fontsize': 30, # was 10
              'xtick.labelsize': 28,
              'ytick.labelsize': 28,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


# mpl.rcParams.update(mpl.rcParamsDefault)
daily_format = '%Y-%m-%d'
datetime_format = '%Y-%m-%d %H:%M:%S'
"""
stage vs volume and area
"""
cont_area_591_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/cont_area.csv'
cont_area_599_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_599/cont_area.csv'
stage_vol_599_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_599/stage_vol.csv'
stage_vol_591_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/stage_vol_new.csv'
cont_area_591_df = pd.read_csv(cont_area_591_file)
cont_area_599_df = pd.read_csv(cont_area_599_file)
stage_vol_599_df = pd.read_csv(stage_vol_599_file)
stage_vol_591_df = pd.read_csv(stage_vol_591_file)
# 
# latexify(fig_width=10, fig_height=6)
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,facecolor='white')
line_1, = ax1.plot(cont_area_591_df['Z'], cont_area_591_df['Area'], '-', lw=2, color='#a70c0b')
# line_2, = ax2.plot(cont_area_599_df['Z'], cont_area_599_df['Area'], '-', lw=2, color='#a70c0b')
line_3,  = ax2.plot(stage_vol_591_df['stage_m'], stage_vol_591_df['total_vol_cu_m'], '-', lw=2)
# line_4, = ax4.plot(stage_vol_599_df['stage_m'], stage_vol_599_df['total_vol_cu_m'], '-', lw=2)
list_ax = [ax1, ax2]
for ax in list_ax:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 30))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid(False)
    ax.set_axis_bgcolor('white')
    # ax.set_axis_off()
ax1.xaxis.set_major_locator(MaxNLocator(nbins=3))
ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
ax2.xaxis.set_major_locator(MaxNLocator(nbins=3))
ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
ax1.tick_params(axis='x',
                which='both',
                labelbottom='off')
# ax3.xaxis.set_major_locator(MaxNLocator(nbins=3))
# ax3.yaxis.set_major_locator(MaxNLocator(nbins=3))
# ax4.xaxis.set_major_locator(MaxNLocator(nbins=3))
# ax4.yaxis.set_major_locator(MaxNLocator(nbins=3))
ax1.set_title('Check dam 591')
# ax2.set_title('Check dam 599')
ax1.set_ylabel(r'Area ($m^2$) ')
ax2.set_ylabel(r"Volume ($m^3$)")
plt.xlabel(r"Stage ($m$)")
# yyl.set_position((-0.1, 0))
# ax1.set_xlim(1.9)
# [__.set_clip_on(False) for __ in plt.gca().get_children()]
plt.savefig('/media/kiruba/New Volume/AGU/poster/agu_checkdam/image/stage_vol_area_591.png',bbox_inches='tight')
plt.show()


# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,facecolor='white')
# line_1, = ax1.plot(cont_area_591_df['Z'], cont_area_591_df['Area'], '-', lw=2, color='#a70c0b')
# line_2, = ax2.plot(cont_area_599_df['Z'], cont_area_599_df['Area'], '-', lw=2, color='#a70c0b')
# line_3,  = ax3.plot(stage_vol_591_df['stage_m'], stage_vol_591_df['total_vol_cu_m'], '-', lw=2)
# line_4, = ax4.plot(stage_vol_599_df['stage_m'], stage_vol_599_df['total_vol_cu_m'], '-', lw=2)
# list_ax = [ax1, ax2, ax3, ax4]
# for ax in list_ax:
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_position(('outward', 20))
#     ax.spines['left'].set_position(('outward', 30))
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_ticks_position('bottom')
#     ax.grid(False)
#     ax.set_axis_bgcolor('white')
# ax1.xaxis.set_major_locator(MaxNLocator(nbins=3))
# ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
# ax2.xaxis.set_major_locator(MaxNLocator(nbins=3))
# ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
# ax3.xaxis.set_major_locator(MaxNLocator(nbins=3))
# ax3.yaxis.set_major_locator(MaxNLocator(nbins=3))
# ax4.xaxis.set_major_locator(MaxNLocator(nbins=3))
# ax4.yaxis.set_major_locator(MaxNLocator(nbins=3))
# ax1.set_title('Check dam 591')
# ax2.set_title('Check dam 599')
# ax1.set_ylabel(r'Area ($m^2$) ')
# ax3.set_ylabel(r"Volume ($m^3$)")
# yyl = plt.xlabel(r"Stage ($m$)")
# yyl.set_position((-0.1, 0))
# # ax1.set_xlim(1.9)
# plt.show()
# raise SystemExit(0)
# colorbrewer
dark2_colors = brewer2mpl.get_map('Dark2', 'Qualitative', 7).mpl_colors
inflow_file = '/media/kiruba/New Volume/ACCUWA_Data/tghalliinflowsbaseflowmonthandindex/tghalli_inflow.csv'
inflow_df = pd.read_csv(inflow_file)
inflow_df.index = pd.to_datetime(inflow_df['Year'], format='%Y')
base_flow_file = '/media/kiruba/New Volume/ACCUWA_Data/tghalliinflowsbaseflowmonthandindex/base_flow.csv'
base_flow_df = pd.read_csv(base_flow_file)
base_flow_df.index = pd.to_datetime(base_flow_df['Period'], format='%b-%Y')
base_flow_df = base_flow_df.resample('A', how=np.sum)
# print base_flow_df.head()
latexify(fig_width=13.2, fig_height=8)
fig,(ax, ax1) = plt.subplots(nrows=2,ncols=1, sharex=True,facecolor='white')
lowess_line = lowess(inflow_df['ML/Year'], inflow_df['Year'])
inflow = ax.bar(inflow_df['Year'], inflow_df['ML/Year'],color='#203a72')
ax.plot(lowess_line[:,0], lowess_line[:,1],'-', lw=3, color= '#23af2b')
base_flow = ax1.bar(base_flow_df.index.year, base_flow_df['Baseflow'], color='#a70c0b')
fig.autofmt_xdate()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_position(('outward', 20))
ax.spines['left'].set_position(('outward', 30))
ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
ax.grid(False)
ax.set_axis_bgcolor('white')
ax.set_xlim(min(inflow_df['Year']), max(inflow_df['Year']))
ax.set_ylim(min(inflow_df['ML/Year']), max(inflow_df['ML/Year']))
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_position(('outward', 20))
ax1.spines['left'].set_position(('outward', 30))
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
ax1.grid(False)
ax1.set_axis_bgcolor('white')
ax1.set_ylabel(r'No of months')
ax.set_ylabel(r"Million Litres/year")
my_locator = MaxNLocator(6)
y_locator = MaxNLocator(3)
x_locator = MaxNLocator(3)
ax1.yaxis.set_major_locator(x_locator)
ax1.xaxis.set_major_locator(my_locator)
ax.yaxis.set_major_locator(y_locator)
ax.legend([inflow,base_flow], [r'Inflow (ML/year)', r"Baseflow (months)"],fancybox=True, loc="upper right")
ax1.yaxis.labelpad=57
# print ax.yaxis.labelpad
# plt.show()
# mpl.rcParams.update(mpl.rcParamsDefault)
file_591 = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/et_infilt_591_w_of.csv'
# file_599 = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_599/daily_wb_599.CSV'
rain_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/ksndmc_rain.csv'
stage_591_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/stage_591.csv'
# stage_599_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_599/water_level.csv'
rain_df = pd.read_csv(rain_file, sep=',', header=0)
rain_df['Date_Time'] = pd.to_datetime(rain_df['Date_Time'], format=datetime_format)
rain_df.set_index(rain_df['Date_Time'], inplace=True)
# sort based on index
rain_df.sort_index(inplace=True)
# drop date time column
rain_df = rain_df.drop('Date_Time', 1)
rain_w_df = rain_df.resample('W-MON', how=np.sum)
rain_w_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/weekly_rain_aral.csv')
# print rain_w_df.tail(10)
# raise SystemExit(0)
rain_df = rain_df.resample('D', how=np.sum)

wb_591 = pd.read_csv(file_591, sep=',', header=0)
# wb_599 = pd.read_csv(file_599, sep=',', header=0)
stage_591_df = pd.read_csv(stage_591_file, sep=',', header=0)
stage_591_df.set_index(pd.to_datetime(stage_591_df['Date'],format=datetime_format),  inplace=True)
# stage_599_df = pd.read_csv(stage_599_file, sep=',', header=0)
# stage_599_df.set_index(pd.to_datetime(stage_599_df['Date'],format=datetime_format),  inplace=True)
wb_591.set_index(pd.to_datetime(wb_591['Date'], format=daily_format), inplace=True)
# wb_599.set_index(pd.to_datetime(wb_599['Date'], format=daily_format), inplace=True)
del wb_591['Date']
# del wb_599['Date']
# missing date time for 599
# with open("/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_599/initial_time.pickle", "rb") as f:
#     initial_time = pickle.load(f)
# with open("/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_599/final_time.pickle", "rb") as f:
#     final_time = pickle.load(f)

# initial_time_591 = initial_time.strftime(daily_format)
# final_time_591 =  final_time.strftime(daily_format)
# missing_data_days = (final_time-initial_time).days

# wb_599 = wb_599[wb_599['infiltration(cu.m)'] > 0]
stage_591_df = stage_591_df.resample('D', how=np.mean)
# stage_599_df = stage_599_df.resample('D', how=np.mean)

def annotate_dim(ax,xyfrom,xyto,text=None):
    ax.annotate("",xyfrom,xyto,arrowprops=dict(arrowstyle='<->', linewidth=1.5))
    # ax.annotate(text, xy=(xyfrom[0]+ timedelta(days=17), xyfrom[1]),xycoords='data', xytext=(-10,-10), textcoords='offset points')


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.itervalues():
        sp.set_visible(False)

latexify(fig_width=15, fig_height=10)
fig, ax1 = plt.subplots(nrows=1,ncols=1, sharex=True, facecolor='white')
# ax1_1 = ax1.twinx()
# ax2_2 = ax2.twinx()
bar_1_1 = ax1.bar(rain_df.index, rain_df['rain (mm)'], 0.45, color='#203a72',alpha=0.85, label = 'Rainfall (mm)')
ax1.invert_yaxis()
for t1 in ax1.get_yticklabels():
    t1.set_color('#203a72')
ax1_1 = ax1.twinx()
ax1_2 = ax1.twinx()
# bar_2_1 = ax2.bar(rain_df.index, rain_df['Rain Collection (mm)'], 0.45, color='#203a72',alpha=0.85, label = 'Rainfall (mm)')
# ax2.invert_yaxis()
# for t1 in ax2.get_yticklabels():
#     t1.set_color('#203a72')
# ax2_1 = ax2.twinx()
bar_1 = ax1_2.bar(wb_591.index, wb_591['infiltration(cu.m)'], 0.45, color='#23530b',alpha=0.85,label=r"\textbf{Infiltration ($m^3/day$}")
line_1, = ax1_2.plot(stage_591_df.index, stage_591_df['stage(m)'], color='#a70c0b',linestyle='-', lw=3, alpha=0.75)
# line_2 = ax2_2.plot(stage_599_df.index, stage_599_df['stage(m)'], color='#a70c0b',linestyle='-', lw=3, alpha=0.75)
# bar_2 = ax2_1.bar(wb_599.index, wb_599['infiltration(cu.m)'], 0.45, color='#23530b',alpha=0.85)
bar_1_2 = ax1_2.bar(wb_591.index, wb_591['Evaporation (cu.m)'], 0.45, color='#a70c0b',alpha=0.85, label=r"\textbf{Evaporation ($m^3/day$)}")
# bar_2_2 = ax2_1.bar(wb_599.index, wb_599['Evaporation (cu.m)'], 0.45, color='#a70c0b',alpha=0.85, label=r"\textbf{Evaporation ($m^3/day$)}")
# bracket = annotate_dim(ax2_1, xyfrom=[initial_time_591,1], xyto=[final_time_591,1], text='Missing Data')
# text = ax2_1.text(initial_time, 2, "Missing Data")
lns = [bar_1_1, bar_1, bar_1_2, line_1]
labs = [r'\textbf{Rainfall ($mm$)}', r"\textbf{Infiltration ($m^3/day$)}", r"\textbf{Evaporation ($m^3/day$)}", r"\textbf{Stage ($m$)}"]
ax1.legend(lns, labs, loc='upper center', fancybox=True, ncol=4, bbox_to_anchor=(0.5, -0.05),prop={'size':30} )
# yyl = plt.ylabel(r"Evaporation/Infiltration ($m^3/day$)")
# yyl.set_position((0.06, 1))
# yyl_1 = ax2.set_ylabel(r'Rainfall($mm$)')
# yyl_1.set_position((yyl_1.get_position()[0], 1))
# fig.text(0.06, 0.5, 'Rainfall (mm)', ha='center', va='center', rotation='vertical')
# plt.figtext(0.95, 0.5, r'Evaporation/Infiltration ($m^3/day)$', ha='center', va='center', rotation='vertical')
ax1.set_title("Check Dam 591")
# ax2_1.set_title("Check Dam 599")
# ax1_2.spines['right'].set_position(('axes', -0.6))
# make_patch_spines_invisible(ax1_2)
# ax1_2.spines['right'].set_visible(True)
ax1_1.yaxis.set_label_position('left')
ax1_1.yaxis.set_ticks_position('left')
for t1 in ax1_1.get_yticklabels():
    t1.set_color('#a70c0b')
ax1_1.set_axis_bgcolor('white')
locator_1 = MaxNLocator(3)
locator_2 = MaxNLocator(3)
locator_1_1 = MaxNLocator(3)
locator_1_2 = MaxNLocator(3)
locator_2_1 = MaxNLocator(3)
locator_2_2 = MaxNLocator(3)
ax1.yaxis.set_major_locator(locator_1)
# ax2.yaxis.set_major_locator(locator_2)
# ax1_1.yaxis.set_major_locator(locator_1_1)
ax1_1.yaxis.set_major_locator(locator_1_2)
# ax2_1.yaxis.set_major_locator(locator_2_1)
# ax2_2.yaxis.set_major_locator(locator_2_2)
ax1_1.spines['top'].set_visible(False)
ax1_1.spines['right'].set_visible(False)
ax1_1.spines['bottom'].set_visible(False)
ax1_1.spines['left'].set_position(('outward', 50))
ax1_1.yaxis.set_ticks_position('left')
ax1_1.xaxis.set_ticks_position('bottom')
ax1_1.tick_params(axis='y', colors='#a70c0b')
ax1_1.spines['left'].set_color('#a70c0b')
# ax1_2.set_ylabel("Stage (m)")
ax1_1.yaxis.label.set_color('#a70c0b')
# ax1.set_ylabel('Rainfall (mm)')
ax1.yaxis.label.set_color('#203a72')
# ax2_2.spines['top'].set_visible(False)
# ax2_2.spines['right'].set_visible(False)
# ax2_2.spines['bottom'].set_visible(False)
# ax2_2.spines['left'].set_position(('outward', 50))
# ax2_2.yaxis.set_ticks_position('left')
# ax2_2.xaxis.set_ticks_position('bottom')
# ax2_2.tick_params(axis='y', colors='#a70c0b')
# ax2_2.spines['left'].set_color('#a70c0b')
# ax2_2.set_ylabel("Stage (m)")
# ax2_2.yaxis.label.set_color('#a70c0b')
# increase tick label size
# left y axis 1
# for tick in ax2.yaxis.get_major_ticks():
#     tick.label.set_fontsize(24)
# # left y axis 2
# for tick in ax1.yaxis.get_major_ticks():
#     tick.label.set_fontsize(24)
# # xaxis
# for tick in ax2.xaxis.get_major_ticks():
#     tick.label.set_fontsize(24)
# for tick in ax2.yaxis.get_major_ticks():
#     tick.label.set_fontsize(24)
# for tick in ax1_1.get_yticklabels():
#     tick.set_fontsize(24)
# for tick in ax2_1.get_yticklabels():
#     tick.set_fontsize(24)
plt.tight_layout()
fig.autofmt_xdate(rotation=90)
plt.savefig('/media/kiruba/New Volume/AGU/poster/agu_checkdam/image/evap_infilt.pdf', dpi=400)
plt.show()
# raise SystemExit(0)
# pie charts
# 591
print wb_599.head()
evap_591 = wb_591['Evaporation (cu.m)'].sum()
infilt_591 = wb_591['infiltration(cu.m)'].sum()
overflow_591 = wb_591['overflow(cu.m)'].sum()
inflow_591 = wb_591['Inflow (cu.m)'].sum()
pumping_591 = wb_591['pumping (cu.m)'].sum()
check_storage_591 = abs((evap_591+infilt_591+overflow_591+pumping_591)-inflow_591)
evap_599 = wb_599['Evaporation (cu.m)'].sum()
infilt_599 = wb_599['infiltration(cu.m)'].sum()
evap_599 = wb_599['Evaporation (cu.m)'].sum()
infilt_599 = wb_599['infiltration(cu.m)'].sum()
overflow_599 = wb_599['overflow(cu.m)'].sum()
inflow_599 = wb_599['Inflow (cu.m)'].sum()
pumping_599 = wb_599['pumping (cu.m)'].sum()
check_storage_599 = abs((evap_599+infilt_599+overflow_599+pumping_599)-inflow_599)
print evap_599
print infilt_599
print overflow_599
print inflow_599
print pumping_599
# latexify(fig_width=13.2, fig_height=8)
latexify(fig_width=15, fig_height=15)
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, facecolor='white' )
fig, ax1 = plt.subplots(facecolor='white')
pie_1, texts_1, autotexts_1 = ax1.pie([evap_591,infilt_591, overflow_591], labels=['E', 'P', 'O' ],
                colors=['#a70c0b', '#23530b', '#377eb8'],
                autopct='%i%%',
                explode=(0,0.1, 0.1),
                startangle=90)
# pie_2, texts_2, autotexts_2 = ax2.pie([evap_599,infilt_599, overflow_599, inflow_599, pumping_599], labels=['E', 'Pe', 'O', 'I', 'Pu' ],
#                 colors=['#a70c0b', '#23530b', '#377eb8', '#984ea3', '#ff7f00'],
#                 autopct='%i%%',
                # explode=(0,0.1, 0.1, 0.1, 0.1),
                # startangle=90)
# ax1.axis('equal')
# ax2.axis('equal')
# plt.tight_layout()
ax1.set_title('Check Dam 591')
# ax2.set_title('Check Dam 599')
ax1.set_axis_bgcolor('white')
# ax2.set_axis_bgcolor('white')
# plt.subplots_adjust(bottom=0.15)
legend = fig.legend([pie_1[0], pie_1[1], pie_1[2]],["Evaporation", "Percolation", "Overflow"],fancybox=True, ncol=3,loc=(0.01,0.02))
# for label in texts_1:
#     label.set_fontsize(56)
# for label in texts_2:
#     label.set_fontsize(56)
# for label in autotexts_1:
#     label.set_fontsize(48)
# for label in autotexts_2:
#     label.set_fontsize(48)
# plt.tight_layout()
plt.savefig('/media/kiruba/New Volume/AGU/poster/agu_checkdam/image/pie_evap_infilt.png')
plt.show()
# raise SystemExit(0)
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
latexify(fig_width=13.2, fig_height=8)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, facecolor='white' )
ax1.scatter(stage_cal_591, inf_cal_591, facecolor='#203a72',marker='o',s=(np.pi*(5**2)))
ax1.plot(stage_cal_new_591, inf_cal_new_591, '#a70c0b',linestyle='-', linewidth=2)
ax1.text(x=0.4, y=.04, s=r'Infiltration rate = ${0:.2f}{{h}}^{{{1:.2f}}}$'.format(popt_591[0], popt_591[1]))
ax2.scatter(stage_cal_599, inf_cal_599, facecolor='#203a72',marker='o',s=(np.pi*(5**2)))
ax2.plot(stage_cal_new_599, inf_cal_new_599, '#a70c0b',linestyle='-', linewidth=2)
ax2.text(x=0.3, y=.04, s=r'Infiltration rate = ${0:.2f}{{h}}^{{{1:.2f}}}$'.format(popt_599[0], popt_599[1]))
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
ax1.set_ylabel(r'Infiltration rate ($m/day$)')
xxl = ax1.set_xlabel(r'Stage ($m$)')
xxl.set_position((1, xxl.get_position()[1]))
# fig.text(0.5, 0.02, 'Stage (m)', ha='center', va='center')
# plt.figtext(0.03, 0.5, r'Infiltration rate ($m/day)$', ha='center', va='center', rotation='vertical')
ax1.set_title("Check Dam 591")
ax2.set_title("Check Dam 599")
ax2.set_ylim(ax1.get_ylim())
locator_x = MaxNLocator(6)
locator_y = MaxNLocator(4)
locator_x_1 = MaxNLocator(6)
locator_y_1 = MaxNLocator(4)
ax1.yaxis.set_major_locator(locator_y)
ax1.xaxis.set_major_locator(locator_x)
ax2.xaxis.set_major_locator(locator_x_1)
ax2.yaxis.set_major_locator(locator_y_1)
# for tick in ax2.yaxis.get_major_ticks():
#     tick.label.set_fontsize(24)
# for tick in ax2.xaxis.get_major_ticks():
#     tick.label.set_fontsize(24)
# for tick in ax1.yaxis.get_major_ticks():
#     tick.label.set_fontsize(24)
# for tick in ax1.xaxis.get_major_ticks():
#     tick.label.set_fontsize(24)
ax2.legend(["Observation", "Prediction"],fancybox=True)
# plt.tight_layout()
plt.savefig('/media/kiruba/New Volume/AGU/poster/agu_checkdam/image/stage_infilt.pdf', dpi=400)
# plt.show()

