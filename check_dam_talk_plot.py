__author__ = 'kiruba'
import matplotlib
# matplotlib.rcsetup.all_backends
matplotlib.use('tkagg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
import brewer2mpl
from matplotlib import cm
from scipy.optimize import curve_fit
import matplotlib as mpl
from math import sqrt
SPINE_COLOR = 'gray'
from matplotlib.ticker import MaxNLocator
# from statsmodels.nonparametric.smoothers_lowess import lowess
import brewer2mpl
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import  matplotlib.patches as mpatches
import scipy.stats as stats
import ccy_classic_lstsqr
from math import exp
import powerlaw
from matplotlib.ticker import FormatStrFormatter
import checkdam.checkdam as cd

# latex parameters
rc('font', **{'family': 'sans-serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=36)

dark_2_colors = brewer2mpl.get_map("Set2", 'Qualitative', 7).mpl_colors


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks

    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)

    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()


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
              'font.family': 'serif',
              'axes.facecolor' : 'white'
    }

    matplotlib.rcParams.update(params)
# matplotlib.rcParams['axes.facecolor'] = 'white'

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


def annotate_dim(ax,xyfrom,xyto,text=None):
    ax.annotate("",xyfrom,xyto,arrowprops=dict(arrowstyle='<->', linewidth=1.5))
    # ax.annotate(text, xy=(xyfrom[0]+ timedelta(days=17), xyfrom[1]),xycoords='data', xytext=(-10,-10), textcoords='offset points')


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.itervalues():
        sp.set_visible(False)
        
file_results_pie = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/summary_check_dam.csv'
results_pie_df = pd.read_csv(file_results_pie, sep=',', header=0)
results_pie_df.set_index(results_pie_df['Check dam no'], inplace=True)
print results_pie_df.head()
# latexify(fig_width=9, fig_height=9)
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, facecolor='white' )
fig= plt.figure(facecolor='white', figsize=(8,6), dpi=100)
ax1 = plt.subplot(131,aspect=True,axisbg='w')
# fig, ax1 = plt.subplots(facecolor='white')
evap_591 = results_pie_df['Evaporation (cu.m)'][591]
infilt_591 = results_pie_df['Infiltration (cu.m)'][591]
overflow_591 = results_pie_df['Overflow (cu.m)'][591]
storage_591 = results_pie_df['Storage (cu.m)'][591]
pie1, text1, autotexts_1 = ax1.pie([evap_591, infilt_591, overflow_591], labels=['E', 'P', 'O'],
                                   colors=dark_2_colors[0:3],
                                   autopct='%i%%'
                                   )
ax1.set_title("Check dam 591")
ax2 = plt.subplot(132,aspect=True,axisbg='w')
evap_599 = results_pie_df['Evaporation (cu.m)'][599]
infilt_599 = results_pie_df['Infiltration (cu.m)'][599]
overflow_599 = results_pie_df['Overflow (cu.m)'][599]
storage_599 = results_pie_df['Storage (cu.m)'][599]
pie2, text2, autotexts_2 = ax2.pie([evap_599, infilt_599, overflow_599], labels=['E', 'P', 'O'],
                                   colors=dark_2_colors[0:3],
                                   autopct='%i%%')
ax2.set_title("Check dam 599")
ax3 = plt.subplot(133,aspect=True,axisbg='w')
evap_634 = results_pie_df['Evaporation (cu.m)'][634]
infilt_634 = results_pie_df['Infiltration (cu.m)'][634]
overflow_634 = results_pie_df['Overflow (cu.m)'][634]
storage_634 = results_pie_df['Storage (cu.m)'][634]
pie3, text3, autotexts_3 = ax3.pie([evap_634, infilt_634, storage_634], labels=['E', 'P', 'S'],
                                   colors=[dark_2_colors[0], dark_2_colors[1], dark_2_colors[3]],
                                   autopct='%i%%',
                                   )
ax3.set_title("Check dam 634")
legend = fig.legend([pie3[0], pie3[1], pie2[2], pie3[2]],["Evaporation", "Percolation", "Overflow", "Storage"],fancybox=True, ncol=2,loc=(0.01,0.02)).draggable()

for label in text1:
    label.set_fontsize(36)
for label in text2:
    label.set_fontsize(36)
for label in text3:
    label.set_fontsize(36)
plt.show()

# raise SystemExit(0)
"""
dry day infiltration vs stage plot
"""
# input file
daily_format = '%Y-%m-%d'
datetime_format = '%Y-%m-%d %H:%M:%S'
# 591
file_591 = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/et_infilt_591_w_of.csv'
rain_a_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/ksndmc_rain.csv'
stage_591_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/stage_591.csv'
rain_df = pd.read_csv(rain_a_file, sep=',', header=0)
rain_df.loc[:, 'Date_Time'] = pd.to_datetime(rain_df['Date_Time'], format=datetime_format)
rain_df.set_index(rain_df['Date_Time'], inplace=True)
# sort based on index
rain_df.sort_index(inplace=True)
# drop date time column
rain_df = rain_df.drop('Date_Time', 1)
rain_df_daily = rain_df.resample('D', how=np.sum)
wb_591 = pd.read_csv(file_591, sep=',', header=0)
stage_591_df = pd.read_csv(stage_591_file, sep=',', header=0)
stage_591_df.set_index(pd.to_datetime(stage_591_df['Date'], format=datetime_format),  inplace=True)
wb_591.set_index(pd.to_datetime(wb_591['Date'], format=daily_format), inplace=True)
# 599
file_599 = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_599/et_infilt_599_w_of.csv'
# stage_599_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_599/stage_599.csv'
wb_599 = pd.read_csv(file_599, sep=',', header=0)
# stage_599_df = pd.read_csv(stage_599_file, sep=',', header=0)
# stage_599_df.set_index(pd.to_datetime(stage_599_df['Date'],format=datetime_format),  inplace=True)
wb_599.set_index(pd.to_datetime(wb_599['Date'], format=daily_format), inplace=True)
# 634
file_634 = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/et_infilt_634_w_of.csv'
# stage_634_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/stage_634.csv'
wb_634 = pd.read_csv(file_634, sep=',', header=0)
# stage_634_df = pd.read_csv(stage_634_file, sep=',', header=0)
# stage_634_df.set_index(pd.to_datetime(stage_634_df['Date'],format=datetime_format),  inplace=True)
wb_634.set_index(pd.to_datetime(wb_634['Date'], format=daily_format), inplace=True)
# dry days df separation
dry_wb_591_df = wb_591.loc[wb_591['status'] == 'N']
dry_wb_599_df = wb_599.loc[wb_599['status'] == 'N']
dry_wb_634_df = wb_634.loc[wb_634['status'] == 'N']
# print dry_wb_591_df.head()
# round decimals two decimal
dry_wb_591_df.loc[:, 'stage(m)'] = cd.myround(dry_wb_591_df['stage(m)'], decimals=2)
dry_wb_599_df.loc[:, 'stage(m)'] = cd.myround(dry_wb_599_df['stage(m)'], decimals=2)
dry_wb_634_df.loc[:, 'stage(m)'] = cd.myround(dry_wb_634_df['stage(m)'], decimals=2)
dry_wb_591_df.loc[:, 'infiltration(cu.m)'] = cd.myround(dry_wb_591_df['infiltration(cu.m)'], decimals=3)
dry_wb_599_df.loc[:, 'infiltration(cu.m)'] = cd.myround(dry_wb_599_df['infiltration(cu.m)'], decimals=3)
dry_wb_634_df.loc[:, 'infiltration(cu.m)'] = cd.myround(dry_wb_634_df['infiltration(cu.m)'], decimals=3)
dry_wb_591_df = dry_wb_591_df.loc[dry_wb_591_df['stage(m)'] > 0.1]
dry_wb_599_df = dry_wb_599_df.loc[dry_wb_599_df['stage(m)'] > 0.1]
dry_wb_634_df = dry_wb_634_df.loc[dry_wb_634_df['stage(m)'] > 0.1]
dry_wb_591_df = dry_wb_591_df.loc[dry_wb_591_df['infiltration(cu.m)'] > 1]
dry_wb_599_df = dry_wb_599_df.loc[dry_wb_599_df['infiltration(cu.m)'] > 1]
dry_wb_634_df = dry_wb_634_df.loc[dry_wb_634_df['infiltration(cu.m)'] > 1]
dry_wb_591_df = dry_wb_591_df.loc[dry_wb_591_df['infiltration(cu.m)'] < 60]
# add month column
dry_wb_591_df.loc[:, 'month'] = dry_wb_591_df.index.month
dry_wb_599_df.loc[:, 'month'] = dry_wb_599_df.index.month
dry_wb_634_df.loc[:, 'month'] = dry_wb_634_df.index.month
# print dry_wb_591_df.head()
"""
# estimate alpha, beta for 591
stage_591_cal = dry_wb_591_df['stage(m)']
inf_591_cal = dry_wb_591_df['infiltration(cu.m)']
log_x_591 = np.log(stage_591_cal)
log_y_591 = np.log(inf_591_cal)
mask_591 = (log_y_591 == log_y_591) & (log_x_591 == log_x_591) #& (log_x_591 > -1.0)
masked_log_x_591 = log_x_591[mask_591]
masked_log_y_591 = log_y_591[mask_591]

# df = pd.DataFrame(masked_log_x_591.values, columns=['x'])
# df['y'] = masked_log_y_591.values
# print df.head()
# raise SystemExit(0)
# print df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/masked_log.csv')
# raise SystemExit(0)
# print masked_log_x_591
# slope_591, intercept_591 = ccy_classic_lstsqr.ccy_classic_lstsqr(masked_log_x_591, masked_log_y_591)
# slope_591, intercept_591 = np.linalg.lstsq(masked_log_x_591.values, masked_log_y_591.values)[0]
slope_591, intercept_591, r_value_591, p_value_591, std_err_591 = stats.linregress(masked_log_x_591,masked_log_y_591)
print slope_591, intercept_591, r_value_591**2, np.exp(slope_591), np.exp(intercept_591)
# raise SystemExit(0)
# print slope_591, intercept_591
# slope_591 = 0.94684
# intercept_591 = 2.36926
alpha_591 = np.exp(intercept_591) # r estimate
beta_591 = slope_591 # r estimate
print alpha_591, beta_591
# estimate alpha, beta for 599
stage_599_cal = dry_wb_599_df['stage(m)']
inf_599_cal = dry_wb_599_df['infiltration(cu.m)']
log_x_599 = np.log(stage_599_cal)
log_y_599 = np.log(inf_599_cal)
mask_599 = (log_y_599 == log_y_599) & (log_x_599 == log_x_599)
masked_log_x_599 = log_x_599[mask_599]
masked_log_y_599 = log_y_599[mask_599]
# df = pd.DataFrame(masked_log_x_599.values, columns=['x'])
# df['y'] = masked_log_y_599.values

# print df.head()
# raise SystemExit(0)
# print df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/masked_log.csv')
slope_599, intercept_599, r_value_599, p_value_599, std_err_599 = stats.linregress(masked_log_x_599,masked_log_y_599)
# slope_599, intercept_599 = ccy_classic_lstsqr.ccy_classic_lstsqr(masked_log_x_599, masked_log_y_599)
print slope_599, intercept_599
alpha_599 = np.exp(intercept_599)
beta_599 = slope_599
print alpha_599, beta_599
# estimate alpha, beta for 634
stage_634_cal = dry_wb_634_df['stage(m)']
inf_634_cal = dry_wb_634_df['infiltration(cu.m)']
log_x_634 = np.log(stage_634_cal)
log_y_634 = np.log(inf_634_cal)
mask_634 = (log_y_634 == log_y_634) & (log_x_634 == log_x_634)
masked_log_x_634 = log_x_634[mask_634]
masked_log_y_634 = log_y_634[mask_634]
slope_634, intercept_634, r_value_634, p_value_634, std_err_634 = stats.linregress(masked_log_x_634,masked_log_y_634)
# slope_634, intercept_634 = ccy_classic_lstsqr.ccy_classic_lstsqr(masked_log_x_634, masked_log_y_634)
print slope_634, intercept_634
alpha_634 = np.exp(intercept_634)
beta_634 = slope_634
print alpha_634, beta_634
# print r_value_591**2, r_value_599**2, r_value_634**2

# create fit 591

def power_func(h, alpha, beta):
    return (alpha*(h**beta))


def linear_func(x, m, c):
    return (m*x) + c


def infil_func(x, m, c):
    return np.exp((m*np.log(x)) + c)


# popt_591, pcov_591 = curve_fit(f=func, xdata=stage_591_cal, ydata=inf_591_cal)
# popt_591_linear, pcov_591_linear = curve_fit(f=linear_func, xdata=masked_log_x_591, ydata=masked_log_y_591)
# print popt_591_linear
# stage_591_pred = np.linspace(min(stage_591_cal), max(stage_591_cal), 50)
# log_inf_591_pred = func(stage_591_pred, alpha_591, beta_591)
# # create fit 599
# # popt_599, pcov_599 = curve_fit(f=func, xdata=stage_599_cal, ydata=inf_599_cal)
# stage_599_pred = np.linspace(min(stage_599_cal), max(stage_599_cal),50)
# inf_599_pred = func(stage_599_pred, alpha_599, beta_599)
# # create fit 634
# # popt_634, pcov_634 = curve_fit(f=func, xdata=stage_634_cal, ydata=inf_634_cal)
# stage_634_pred = np.linspace(min(stage_634_cal), max(stage_634_cal), 50)
# inf_634_pred = func(stage_634_pred, alpha_634, beta_634)

log_stage_591_pred = np.linspace(min(masked_log_x_591), max(masked_log_x_591), 50)
log_stage_599_pred = np.linspace(min(masked_log_x_599), max(masked_log_x_599), 50)
log_stage_634_pred = np.linspace(min(masked_log_x_634), max(masked_log_x_634), 50)
print min(masked_log_x_591)
print max(masked_log_x_591)
print log_stage_591_pred
# stage_591_pred = np.linspace(min(stage_591_cal), max(stage_591_cal), 50)
log_inf_591_pred = linear_func(log_stage_591_pred, slope_591, intercept_591)
log_inf_599_pred = linear_func(log_stage_599_pred, slope_599, intercept_599)
log_inf_634_pred = linear_func(log_stage_634_pred, slope_634, intercept_634)
# inf_591_pred = np.exp(log_inf_591_pred)
# inf_599_pred = np.exp(log_inf_599_pred)
# inf_634_pred = np.exp(log_inf_634_pred)
# stage_591_pred = np.exp(log_stage_591_pred)
# stage_599_pred = np.exp(log_stage_599_pred)
# stage_634_pred = np.exp(log_stage_634_pred)

# print max(stage_599_pred)
# print max(inf_599_pred)
# cmap_591, norm_591 = mpl.colors.from_levels_and_colors([1, 2, 5, 7, 9, 11, 13], ["#7fc97f", "#beaed4","#fdc086","#ffff99","#386cb0", "#f0027f"])
# cmap_599, norm_599 = mpl.colors.from_levels_and_colors([1, 2, 5, 7, 9, 11, 13], ["#7fc97f", "#beaed4","#fdc086","#ffff99","#386cb0", "#f0027f"])
# cmap_634, norm_634 = mpl.colors.from_levels_and_colors([1, 2, 5, 7, 9, 11, 13], ["#7fc97f", "#beaed4","#fdc086","#ffff99","#386cb0", "#f0027f"])
fig, (ax_1, ax_2, ax_3) = plt.subplots(nrows=1, ncols=3, facecolor='white')
ax_1.plot(masked_log_x_591, masked_log_y_591, 'bo')
ax_2.plot(masked_log_x_599, masked_log_y_599, 'bo')
ax_3.plot(masked_log_x_634, masked_log_y_634, 'bo')
ax_1.plot(log_stage_591_pred, log_inf_591_pred, 'r-')
ax_1.set_title('m = %0.02f , c = %0.02f' % (slope_591, intercept_591))
ax_2.set_title('m = %0.02f , c = %0.02f' % (slope_599, intercept_599))
ax_3.set_title('m = %0.02f , c = %0.02f' % (slope_634, intercept_634))
ax_2.plot(log_stage_599_pred, log_inf_599_pred, 'r-')
ax_3.plot(log_stage_634_pred, log_inf_634_pred, 'r-')
plt.show()

stage_591_pred = np.linspace(0.1, max(stage_591_cal), 50)
stage_599_pred = np.linspace(0.1, max(stage_599_cal), 50)
stage_634_pred = np.linspace(0.1, max(stage_634_cal), 50)
inf_591_pred = power_func(stage_591_pred, alpha_591, beta_591)
inf_599_pred = power_func(stage_599_pred, alpha_599, beta_599)
inf_634_pred = power_func(stage_634_pred, alpha_634, beta_634)

fig, (ax_1, ax_2, ax_3) = plt.subplots(nrows=1, ncols=3, facecolor='white')
scatter_591 = ax_1.scatter(stage_591_cal, inf_591_cal, facecolor='k', marker='o', s=(np.pi*(3**2)))
ax_1.plot(stage_591_pred, power_func(stage_591_pred, alpha_591, beta_591), 'r-')
ax_1.text(x=0.02, y=55, s=r'Percolation = ${0:.2f}{{h}}^{{{1:.2f}}}$'.format(alpha_591, beta_591))
ax_1.set_xlim(0, 2)
ax_1.set_ylim(0, 60)
ax_2.scatter(stage_634_cal, inf_634_cal, facecolor='k', marker='o', s=(np.pi*(3**2)))
ax_2.plot(stage_634_pred, inf_634_pred, 'r-')
ax_2.text(x=0.02, y=9, s=r'Percolation = ${0:.2f}{{h}}^{{{1:.2f}}}$'.format(alpha_634, beta_634))
ax_2.set_xlim(0, 0.6)
ax_2.set_ylim(0, 10)
ax_3.scatter(stage_599_cal, inf_599_cal, facecolor='k', marker='o', s=(np.pi*(3**2)))
ax_3.plot(stage_599_pred, inf_599_pred, 'r-')
ax_3.text(x=0.07, y=27, s=r'Percolation = ${0:.2f}{{h}}^{{{1:.2f}}}$'.format(alpha_599, beta_599))
ax_3.set_xlim(0, 1)
ax_3.set_ylim(0, 30)
ax_2.set_xlabel(r'\textbf{Stage} (m)')
# xxl.set_position((-0.1, 0))
ax_1.set_ylabel(r'\textbf{Percolation} ($m^{3}$)')
# disable additional splines
ax_1.spines['top'].set_visible(False)
ax_1.spines['right'].set_visible(False)
ax_1.yaxis.set_ticks_position('left')
ax_1.xaxis.set_ticks_position('bottom')
ax_2.spines['top'].set_visible(False)
ax_2.spines['right'].set_visible(False)
ax_2.yaxis.set_ticks_position('left')
ax_2.xaxis.set_ticks_position('bottom')
ax_1.spines['bottom'].set_position(('outward', 20))
ax_1.spines['left'].set_position(('outward', 30))
ax_2.spines['bottom'].set_position(('outward', 20))
ax_2.spines['left'].set_position(('outward', 30))
ax_3.spines['top'].set_visible(False)
ax_3.spines['right'].set_visible(False)
ax_3.yaxis.set_ticks_position('left')
ax_3.xaxis.set_ticks_position('bottom')
ax_3.spines['bottom'].set_position(('outward', 20))
ax_3.spines['left'].set_position(('outward', 30))

# reduce no of ticks
ax_1_x_locator = MaxNLocator(3)
ax_1_y_locator = MaxNLocator(3)
ax_1.xaxis.set_major_locator(ax_1_x_locator)
ax_1.yaxis.set_major_locator(ax_1_y_locator)
ax_2_x_locator = MaxNLocator(3)
ax_2_y_locator = MaxNLocator(4)
ax_2.xaxis.set_major_locator(ax_2_x_locator)
ax_2.yaxis.set_major_locator(ax_2_y_locator)
ax_3_x_locator = MaxNLocator(3)
ax_3_y_locator = MaxNLocator(3)
ax_3.xaxis.set_major_locator(ax_3_x_locator)
ax_3.yaxis.set_major_locator(ax_3_y_locator)
# set title
ax_1.set_title('Check dam 591')
ax_2.set_title('Check dam 634')
ax_3.set_title('Check dam 599')

# fig.subplots_adjust(right=0.8)
# cbar = fig.colorbar(scatter_591, )
# cbar.ax.set_ylabel('Month', rotation=270)
# cbar.ax.get_yaxis().set_ticks([])
# for j, lab in enumerate(["1", "2", "5", "7", "9", "11", "12"]):
#     print j, lab
#     cbar.ax.text(.5, (1*j + 1)/14.0, lab, ha='center', va='center')
# cbar.ax.get_yaxis().labelpad = 30

# for X, Y, Z in zip(stage_591_cal, inf_591_cal, dry_wb_591_df.month):
#     ax_1.annotate('{}'.format(Z), xy=(X, Y), xytext=(-5,5), ha='right', textcoords='offset points')
#
plt.show()
print dry_wb_591_df['month'].unique()
raise SystemExit(0)
file_results_pie = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/summary_check_dam.csv'
results_pie_df = pd.read_csv(file_results_pie, sep=',', header=0)
results_pie_df.set_index(results_pie_df['Check dam no'], inplace=True)
# print results_pie_df.head()
fig_1, ax = plt.subplots(1, figsize=(10, 5), facecolor='white')
bar_width = 1
bar_l = [i*2 for i in range(len(results_pie_df['Percentage of E']))]
# print bar_l
tick_pos = [i+(bar_width/2.0) for i in bar_l]
# print tick_pos
totals = results_pie_df['Inflow (cu.m)']
evap = [i /j *100 for i, j in zip(results_pie_df['Evaporation (cu.m)'], totals)]
overflow = [i /j *100 for i, j in zip(results_pie_df['Overflow (cu.m)'], totals)]
infil = [i /j *100 for i, j in zip(results_pie_df['Infiltration (cu.m)'], totals)]
# '#008000', '#FF0000'
# '#3C5F5A'
ax.bar(bar_l, evap, label='Evaporation', alpha=0.9, color='#019600', width=bar_width, edgecolor='white')
ax.bar(bar_l, overflow, bottom=evap, label='Overflow', alpha=0.9, color='#3C5F5A', width=bar_width, edgecolor='white')
ax.bar(bar_l, infil, bottom=np.array(evap) + np.array(overflow), label='Percolation', alpha=0.9, color='#219AD8', width=bar_width, edgecolor='white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_label_position('left')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_label_position('bottom')
ax.xaxis.set_ticks_position('bottom')
# ax.bar(bar_l, results_pie_df['Percentage of E']*100, label='Evaporation', alpha=0.9, color='#019600', width=bar_width, edgecolor='white')
# ax.bar(bar_l, results_pie_df['Percentage of Overflow']*100, bottom=results_pie_df['Percentage of E']*100, label='Overflow', alpha=0.9, color='#3C5F5A', width=bar_width, edgecolor='white')
# ax.bar(bar_l, results_pie_df['Percentage of Infil']*100,bottom=np.array(results_pie_df['Percentage of E']*100)+ np.array(results_pie_df['Percentage of Overflow']*100) ,label='Percolation', alpha=0.9, color='#219AD8', width=bar_width, edgecolor='white')
plt.xticks(tick_pos, ['634', '591', '599'])
ax.set_ylabel(r"Percentage")
ax.set_xlabel("")
ax.set_xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
plt.ylim(-10, 110)
plt.legend(ncol=3).draggable()
plt.show(fig_1)
# raise SystemExit(0)


del wb_591['Date']
wb_591['Inflow (cu.m)'][wb_591['Inflow (cu.m)'] < 0] = 0
width = 300.0/len(wb_591.index)
# print len(wb_591.index)
# print width

# reduced flow due to check dam
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, facecolor='white')
bar_inflow = ax1.bar(wb_591.index, wb_591['Inflow (cu.m)'], width=[(wb_591.index[j+1]-wb_591.index[j]).days for j in range(len(wb_591.index)-1)] + [0], alpha=0.5, color='r')
bar_outflow = ax2.bar(wb_591.index, wb_591['overflow(cu.m)'], width=[(wb_591.index[j+1]-wb_591.index[j]).days for j in range(len(wb_591.index)-1)] + [0], alpha=0.5, color='g')
# bar_inflow = ax1.bar(wb_591.index, wb_591['Inflow (cu.m)'], width=[x.days for x in wb_591.duration], alpha=0.5, color='r')
# bar_outflow = ax2.bar(wb_591.index, wb_591['overflow(cu.m)'], width=[x.days for x in wb_591.duration], alpha=0.5, color='g')

ax1.set_ylim([0, 1800])
ax2.set_ylim([0, 1800])
month_locator = mdates.MonthLocator(interval=3)
xfmt = mdates.DateFormatter('%b-%y')
ax1.xaxis.set_major_locator(month_locator)
ax1.xaxis.set_major_formatter(xfmt)
ax2.xaxis.set_major_locator(month_locator)
ax2.xaxis.set_major_formatter(xfmt)
# ax2.set_yticks([])
ax2.yaxis.set_label_position('right')
ax2.yaxis.set_ticks_position('right')
for t1 in ax1.get_yticklabels():
    t1.set_color('r')
for t1 in ax2.get_yticklabels():
    t1.set_color('g')
# display only 3 ticks
locator_1 = MaxNLocator(3)
locator_2 = MaxNLocator(3)
ax1.yaxis.set_major_locator(locator_1)
ax2.yaxis.set_major_locator(locator_2)
ax1.set_title('Flow with out check dam')
ax2.set_title('Flow due to check dam')
plt.show()

# raise SystemExit(0)
# print 'width = %s'  %width
# wb_591 = wb_591[wb_591['Inflow (cu.m)'] > 0]
# wb_591 = wb_591[:'2014-12-31']
# rain_df = rain_df[:"2014-12-31"]
# print wb_591.head()
# stage_591_df = stage_591_df.resample('D', how=np.mean)
# rainfall intensity
# print rain_df.head()
# hourly rainfall aggregate
rain_df_hourly = rain_df.resample('H', how=np.sum)
# print rain_df_hourly.head()
# amount of rainfall for triggering first inflow event
# print(rain_df_daily['rain (mm)'][:"2014-08-20"].sum())
# list of inflow events
# inflow_events = wb_591[wb_591['Inflow (cu.m)'] > 0].index
# max hourly intensity on a day
max_daily_rain_intensity_mm_hr = rain_df_hourly.resample('D', how=np.max)
# inflow vs daily max intensity

intensity_df = max_daily_rain_intensity_mm_hr['2014-05-15':]
# print len(intensity_df)
# fig = plt.figure(facecolor='white')
# plt.scatter(intensity_df['rain (mm)'], wb_591['Inflow (cu.m)'])
# plt.ylabel("Inflow (cu.m)")
# plt.xlabel("Rainfall Intensity (mm/hr)")
# plt.show()
intensity_df.columns.values[0] = 'intensity (mm/hr)'
# print intensity_df.head()
# print wb_591.head()
intensity_df_select = wb_591.join(intensity_df, how='right')
# print(intensity_df_select.head())
intensity_df_select = intensity_df_select[intensity_df_select['intensity (mm/hr)'] > 2.0]
# print intensity_df_select.head()
intensity_df_select['log_intensity'] = np.log(intensity_df_select['intensity (mm/hr)'])
intensity_df_select['log_inflow'] = np.log(intensity_df_select['Inflow (cu.m)'])
intensity_df_select = intensity_df_select[(intensity_df_select['log_intensity'] > 0) & (intensity_df_select['log_inflow'] > 0)]
print intensity_df_select.head()

log_x = intensity_df_select['log_intensity']
log_y = intensity_df_select['log_inflow']

# log_x = np.log(intensity_cal)
# log_y = np.log(inflow_cal)

mask = ~np.isnan(log_x) & ~np.isnan(log_y)
masked_log_x = log_x[mask]
masked_log_y = log_y[mask]
print masked_log_y
slope, intercept, r_value, p_value, stderr = stats.linregress(masked_log_x, masked_log_y)
# slope, intercept = ccy_classic_lstsqr.ccy_classic_lstsqr(masked_log_x, masked_log_y)
print slope, intercept, r_value, p_value, stderr
intensity_cal_new = np.linspace(min(log_x), max(log_x), 100)
inflow_cal_new = slope*intensity_cal_new + intercept
fig = plt.figure(facecolor='white')
plt.scatter(masked_log_x, masked_log_y)
plt.plot(intensity_cal_new, inflow_cal_new, 'r-')
plt.ylabel(r"log-Inflow ($m^{3}$)")
plt.xlabel(r"log-Rainfall Intensity ($mm hr^{-1}$)")
plt.text(0.75, 7,r"$R^{2} = %0.02f$ \\ p-value = %0.02f" %(r_value, p_value))
# plt.title("Log Inflow vs Log Max daily intensity")
plt.show()
"""
"""
infiltration rate vs stage
"""

dry_wb_591_df.loc[:, 'new_infiltration_rate(m)'] = dry_wb_591_df['infiltration(cu.m)']/dry_wb_591_df['ws_area(sq.m)']
# dry_wb_599_df['infiltration_rate(m)'] = dry_wb_599_df['infiltration(cu.m)']/dry_wb_599_df['ws_area(sq.m)']
# dry_wb_634_df['infiltration_rate(m)'] = dry_wb_634_df['infiltration(cu.m)']/dry_wb_634_df['ws_area(sq.m)']
print "Average Infiltration rate "
print dry_wb_591_df['infiltration_rate(m)'].mean()
stage_cal_591 = dry_wb_591_df['stage(m)'].values
infilt_rate_cal_591 = dry_wb_591_df['new_infiltration_rate(m)'].values
stage_cal_599 = dry_wb_599_df['stage(m)'].values
infilt_rate_cal_599 = dry_wb_599_df['infiltration_rate(m)'].values
stage_cal_634 = dry_wb_634_df['stage(m)'].values
infilt_rate_cal_634 = dry_wb_634_df['infiltration_rate(m)'].values
dry_wb_591_df.loc[dry_wb_591_df['month'] == 1] = 13
dry_wb_591_df.loc[dry_wb_591_df['month'] == 2] = 14
print dry_wb_591_df['month'].unique()
print dry_wb_599_df['month'].unique()
print dry_wb_634_df['month'].unique()
# raise SystemExit(0)
fig, (ax_1, ax_2, ax_3) = plt.subplots(nrows=1, ncols=3, facecolor='white')
scatter_591 = ax_1.scatter(stage_cal_591, infilt_rate_cal_591, facecolor='k', marker='o', s=(np.pi*(3**2)))
scatter_599 = ax_2.scatter(stage_cal_599, infilt_rate_cal_599, facecolor='k', marker='o', s=(np.pi*(3**2)))
scatter_634 = ax_3.scatter(stage_cal_634, infilt_rate_cal_634, facecolor='k', marker='o', s=(np.pi*(3**2)))
ax_1.set_xlim(0, 2)
ax_1.set_title("591")
ax_2.set_title("599")
ax_3.set_title("634")
plt.show()
print(dry_wb_591_df['infiltration_rate(m)'].head(20))

norm = matplotlib.colors.Normalize(vmin=5, vmax=14, clip=False)

fig, (ax_1, ax_2, ax_3) = plt.subplots(nrows=1, ncols=3, facecolor='white', sharey=True)
cmap = cm.get_cmap('Spectral_r')
scatter_591 = ax_1.scatter(stage_cal_591, infilt_rate_cal_591, c=dry_wb_591_df['month'], cmap=cmap, norm=norm, edgecolor='None', s=(np.pi*(5**2)))
scatter_599 = ax_2.scatter(stage_cal_599, infilt_rate_cal_599, c=dry_wb_599_df['month'], cmap=cmap, norm=norm, edgecolor='None', s=(np.pi*(5**2)))
scatter_634 = ax_3.scatter(stage_cal_634, infilt_rate_cal_634, c=dry_wb_634_df['month'], cmap=cmap, norm=norm, edgecolor='None', s=(np.pi*(5**2)))
ax_1.set_title('591')
ax_1.set_xlim(0, 2)
ax_1.set_ylim(0, 0.08)
ax_2.set_title('599')
ax_3.set_title('634')
# locators
locator_x_591 = MaxNLocator(3)
locator_x_599 = MaxNLocator(3)
locator_x_634 = MaxNLocator(3)
locator_y_591 = MaxNLocator(4)
# locator_y_599 = MaxNLocator(4)
# locator_y_634 = MaxNLocator(4)
# xaxis
ax_1.xaxis.set_major_locator(locator_x_591)
ax_2.xaxis.set_major_locator(locator_x_599)
ax_3.xaxis.set_major_locator(locator_x_634)
# yaxis
ax_1.yaxis.set_major_locator(locator_y_591)
# ax_2.yaxis.set_major_locator(locator_y_599)
# ax_3.yaxis.set_major_locator(locator_y_634)
# set xaxis, yaxis labels
ax_2.set_xlabel(r"Stage (m)")
ax_1.set_ylabel(r"Percolation rate ($m \ day^{-1}$)", labelpad=20)
# colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.4, 0.04, 0.5])
cbar = fig.colorbar(scatter_591, cax=cbar_ax)
print cbar.ax.get_yticks()
cbar.ax.get_yaxis().set_ticks([0,  0.333, 0.666, 1])
cbar.ax.get_yaxis().set_ticklabels(['May-14', 'Aug-14', 'Nov-14', 'Feb-15'])

# for j, lab in enumerate([5, ])
plt.show()
#
# cmap_591, norm_591 = mpl.colors.from_levels_and_colors([1, 3, 5, 9, 11, 13], ["maroon", "red","mediumpurple", "lightblue", "indigo"])
# cmap_599, norm_599 = mpl.colors.from_levels_and_colors([1, 3, 5, 9, 11, 13], ["maroon", "red","mediumpurple", "lightblue", "indigo"])
# cmap_634, norm_634 = mpl.colors.from_levels_and_colors([1, 3, 5, 9, 11, 13], ["maroon", "red","mediumpurple", "lightblue", "indigo"])

# cmap_599, norm_599 = mpl.colors.from_levels_and_colors([5, 6, 8, 9, 10, 11, 12, 1, 2], ["#7fc97f", "#beaed4","#fdc086","#ffff99","#386cb0", "#f0027f"])
# cmap_634, norm_634 = mpl.colors.from_levels_and_colors([5, 6, 8, 9, 10, 11, 12, 1, 2], ["#7fc97f", "#beaed4","#fdc086","#ffff99","#386cb0", "#f0027f"])
#
# fig, (ax_1, ax_2, ax_3) = plt.subplots(nrows=1, ncols=3, facecolor='white')
# # fig = plt.figure()
# scatter_591 = ax_1.scatter(stage_cal_591, infilt_rate_cal_591, c=dry_wb_591_df['month'], cmap=cmap_591, norm=norm_591, edgecolor='None',s=(np.pi*(5**2)) )
# scatter_599 = ax_2.scatter(stage_cal_599, infilt_rate_cal_599, c=dry_wb_599_df['month'], cmap=cmap_599, norm=norm_599, edgecolor='None',s=(np.pi*(5**2)) )
# scatter_634 = ax_3.scatter(stage_cal_634, infilt_rate_cal_634, c=dry_wb_634_df['month'], cmap=cmap_634, norm=norm_634, edgecolor='None',s=(np.pi*(5**2)) )
# ax_1.set_title('591')
# ax_1.set_xlim(0, 2)
# ax_2.set_title('599')
# ax_3.set_title('634')
# cbar = fig.colorbar(scatter_591)
# plt.show()

#  stage vs infil plot based days from inflow


def find_previous_inflow_date(df, inflow_dates):
    """
    Calculates no of days from inflow event for dates in df

    :param df: Input df
    :param inflow_dates:datetime index of inflow pandas dataframe
    :return:
    """
    # insert_dummy_columns
    df['days_from_inflow'] = 0
    for date in df.index:
        deltas = inflow_dates - date
        days_from_inflow = np.max([n for n in deltas.days if n < 0])
        df.loc[date, 'days_from_inflow'] = np.abs(days_from_inflow)
    return df

# print len(inflow_days_591_df.index)

inflow_days_591_df = wb_591.loc[wb_591['Inflow (cu.m)'] > 1]
find_previous_inflow_date(dry_wb_591_df, inflow_days_591_df.index)
inflow_days_599_df = wb_599.loc[wb_599['Inflow (cu.m)'] > 1]
find_previous_inflow_date(dry_wb_599_df, inflow_days_599_df.index)
inflow_days_634_df = wb_634.loc[wb_634['Inflow (cu.m)'] > 1]
find_previous_inflow_date(dry_wb_634_df, inflow_days_634_df.index)
# print inflow_days_591_df.head(10)
# print dry_wb_591_df.head(10)

fig, (ax_1, ax_2, ax_3) = plt.subplots(nrows=1, ncols=3, facecolor='white', sharey=True)
# fig = plt.figure()
cmap = cm.get_cmap('Spectral_r')
scatter_591 = ax_1.scatter(stage_cal_591, infilt_rate_cal_591, c=dry_wb_591_df['days_from_inflow'], cmap=cmap, edgecolor='None',s=(np.pi*(5**2)) )
scatter_599 = ax_2.scatter(stage_cal_599, infilt_rate_cal_599, c=dry_wb_599_df['days_from_inflow'], cmap=cmap, edgecolor='None',s=(np.pi*(5**2)) )
scatter_634 = ax_3.scatter(stage_cal_634, infilt_rate_cal_634, c=dry_wb_634_df['days_from_inflow'], cmap=cmap, edgecolor='None',s=(np.pi*(5**2)) )
ax_1.set_title('591')
ax_1.set_xlim(0, 2)
ax_1.set_ylim(0, 0.08)
ax_2.set_title('599')
ax_3.set_title('634')
cbar = fig.colorbar(scatter_591) #, format='%i')
# locators
locator_x_591 = MaxNLocator(3)
locator_x_599 = MaxNLocator(3)
locator_x_634 = MaxNLocator(3)
locator_y_591 = MaxNLocator(4)
# locator_y_599 = MaxNLocator(4)
# locator_y_634 = MaxNLocator(4)
# xaxis
ax_1.xaxis.set_major_locator(locator_x_591)
ax_2.xaxis.set_major_locator(locator_x_599)
ax_3.xaxis.set_major_locator(locator_x_634)
# ax_1.xaxis.set_major_locator(locator_x_591)
ax_2.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax_3.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
# yaxis
# ax_1.yaxis.set_major_locator(locator_y_591)
ax_2.set_xlabel(r"Stage (m)")
ax_1.set_ylabel(r"Percolation rate ($m \ day^{-1}$)", labelpad=20)
plt.show()
raise SystemExit(0)
"""
fig , ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, facecolor='white')
bbox = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
width, height = bbox.width, bbox.height
# print width
bar_rain = ax1.bar(rain_df.index, rain_df['rain (mm)'], width=[(rain_df.index[j+1]-rain_df.index[j]).days for j in range(len(rain_df.index)-1)] + [30], color=dark_2_colors[2],alpha=0.9,label = 'Rainfall (mm)')
ax1.invert_yaxis()
for t1 in ax1.get_yticklabels():
    t1.set_color(dark_2_colors[2])
ax1_1 = ax1.twinx()
ax1_2 = ax1.twinx()
ax1_3 = ax1.twinx()
wb_591['Inflow (cu.m)'][wb_591['Inflow (cu.m)'] < 0] = 0
line_stage, = ax1_1.plot(stage_591_df.index, stage_591_df['stage(m)'], color='#e41a1c',linestyle='-', lw=1.5, alpha=0.85)
bar_overflow = ax1_3.bar(wb_591.index, wb_591['overflow(cu.m)'], width=[(wb_591.index[j+1]-wb_591.index[j]).days for j in range(len(wb_591.index)-1)] + [30], color=dark_2_colors[5], alpha=1)
bar_inflow = ax1_2.bar(wb_591.index, wb_591['Inflow (cu.m)'], width=[(wb_591.index[j+1]-wb_591.index[j]).days for j in range(len(wb_591.index)-1)] + [30],color=dark_2_colors[4],alpha=1)
# lns = [bar_rain, line_stage, bar_inflow]
# labs = [r'\textbf{Rainfall ($mm$)}', r"\textbf{Stage ($m$)}", r"\textbf{Inflow ($m^3$)}"]
# ax1.legend(lns, labs,prop={'size':30} ).draggable()
ax1.set_title("Check dam 591")
for t1 in ax1_1.get_yticklabels():
    t1.set_color('#e41a1c')
for t1 in ax1_2.get_yticklabels():
    t1.set_color(dark_2_colors[4])
# set ticks for stage in left
ax1_1.yaxis.set_label_position('left')
ax1_1.yaxis.set_ticks_position('left')
ax1_1.spines['top'].set_visible(False)
ax1_1.spines['right'].set_visible(False)
ax1_1.spines['bottom'].set_visible(False)
ax1_1.spines['left'].set_position(('outward', 80))
ax1_1.tick_params(axis='y', colors='#e41a1c')
ax1_1.spines['left'].set_color('#e41a1c')
# for inflow and rain change ticks color
ax1_2.tick_params(axis='y', colors=dark_2_colors[4])
ax1_2.spines['right'].set_color(dark_2_colors[4])
# display only 3 ticks
locator_1 = MaxNLocator(3)
locator_1_1 = MaxNLocator(3)
locator_1_2 = MaxNLocator(3)
month_locator = mdates.MonthLocator(interval=2)
xfmt = mdates.DateFormatter('%b-%Y')
ax1.yaxis.set_major_locator(locator_1)
ax1_1.yaxis.set_major_locator(locator_1_1)
ax1_2.yaxis.set_major_locator(locator_1_2)
ax1_2.xaxis.set_major_locator(month_locator)
ax1_2.xaxis.set_major_formatter(xfmt)
fig.autofmt_xdate(rotation=90)
plt.show()
# raise  SystemExit(0)
"""
"""
# stack plot
# print rain_df.tail()
# print wb_591.tail()
fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, facecolor='white')
bar_rain = ax1.bar(rain_df.index, rain_df['rain (mm)'], width=[(rain_df.index[j+1]-rain_df.index[j]).days for j in range(len(rain_df.index)-1)] + [30], color=dark_2_colors[2],alpha=0.9,label = 'Rainfall (mm)')
ax1.set_ylim([0, 300])
ax1.invert_yaxis()
for t1 in ax1.get_yticklabels():
    t1.set_color(dark_2_colors[2])
ax1_1 = ax1.twinx()
label_list = ['Overflow', 'Inflow', "Rainfall"]
inflow = ax1_1.fill_between(wb_591.index, wb_591['overflow(cu.m)'].cumsum(), color="none", edgecolor='#0000FF')
outflow = ax1_1.fill_between(wb_591.index, wb_591['Inflow (cu.m)'].cumsum(), color="none", edgecolor='#FF0000')
# stack = ax1_1.stackplot(wb_591.index, wb_591['overflow(cu.m)'].cumsum(), wb_591['Inflow (cu.m)'].cumsum(), colors=['#FFFFFF', '#000000'], alpha=0.5, zorder=-32 )
# display only 3 ticks
locator_1 = MaxNLocator(3)
locator_1_1 = MaxNLocator(3)
month_locator = mdates.MonthLocator(interval=3)
xfmt = mdates.DateFormatter('%b-%Y')
ax1.yaxis.set_major_locator(locator_1)
ax1_1.yaxis.set_major_locator(locator_1_1)
ax1_1.xaxis.set_major_locator(month_locator)
ax1_1.xaxis.set_major_formatter(xfmt)
ax1_1.set_xlim([min(rain_df.index), max(wb_591.index)])
ax1_1.legend([mpatches.Patch(facecolor='#FFFFFF',edgecolor='#0000FF', alpha=0.5), mpatches.Patch(facecolor='#FFFFFF',edgecolor='#FF0000', alpha=0.5),mpatches.Patch(color=dark_2_colors[2], alpha=0.9)], label_list).draggable()
# ax1_1.legend([mpatches.Patch(color='#FFFFFF', alpha=0.5), mpatches.Patch(color='#000000', alpha=0.5)], label_list).draggable()
ax1_1.set_ylabel(r"Flow ($m^{3} \, day^{-1}$)")
ax1.set_ylabel(r"Rainfall ($mm \, day^{-1}$)")
# ax1_1.set_title("Check dam 591")
#
fig.autofmt_xdate(rotation=90)
# # print evap
plt.show()
raise  SystemExit(0)

"""

print max_daily_rain_intensity_mm_hr.head()
print '2014-08-20'
print max_daily_rain_intensity_mm_hr['rain (mm)']['2014-08-20']
print rain_df_daily['rain (mm)']['2014-08-20']
print '2014-08-21'
print max_daily_rain_intensity_mm_hr['rain (mm)']['2014-08-21']
print rain_df_daily['rain (mm)']['2014-08-21']
print '2014-10-08'
print max_daily_rain_intensity_mm_hr['rain (mm)']['2014-10-08']
print rain_df_daily['rain (mm)']['2014-10-08']
print '2014-10-09'
print max_daily_rain_intensity_mm_hr['rain (mm)']['2014-10-09']
print rain_df_daily['rain (mm)']['2014-10-09']
# fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, facecolor='white')
# bar_rain = ax.bar(rain_df_hourly.index, rain_df_hourly['rain (mm)'], width=[(rain_df_hourly.index[j+1]-rain_df_hourly.index[j]).days for j in range(len(rain_df_hourly.index)-1)] + [24])
# ax.invert_yaxis()
# ax_1 = ax.twinx()
# # stage_plot = ax_1.plot(stage_591_df.index, stage_591_df['stage(m)'], 'r-')
# inflow_plot = ax_1.bar(wb_591.index, wb_591['Inflow (cu.m)'], color='r')
# fig.autofmt_xdate(rotation=90)
# plt.show()
#intensity vs inflow outflow plot
fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, facecolor='white')
bar_rain = ax1.bar(max_daily_rain_intensity_mm_hr.index, max_daily_rain_intensity_mm_hr['rain (mm)'], width=[(max_daily_rain_intensity_mm_hr.index[j+1]-max_daily_rain_intensity_mm_hr.index[j]).days for j in range(len(max_daily_rain_intensity_mm_hr.index)-1)] + [30], color='#0000FF',alpha=0.7,label= 'Rainfall (mm)')
ax1.set_ylim([0, 150])
ax1.invert_yaxis()
for t1 in ax1.get_yticklabels():
    t1.set_color(dark_2_colors[2])
ax1_1 = ax1.twinx()
label_list = ['Overflow', 'Inflow', "Rainfall"]
stack = ax1_1.stackplot(wb_591.index, wb_591['overflow(cu.m)'], wb_591['Inflow (cu.m)'], colors=['#008000', '#FF0000'], alpha=0.5)
# display only 3 ticks
locator_1 = MaxNLocator(3)
locator_1_1 = MaxNLocator(3)
month_locator = mdates.MonthLocator(interval=3)
xfmt = mdates.DateFormatter('%b-%Y')
ax1.yaxis.set_major_locator(locator_1)
ax1_1.yaxis.set_major_locator(locator_1_1)
ax1_1.xaxis.set_major_locator(month_locator)
ax1_1.xaxis.set_major_formatter(xfmt)
ax1_1.set_xlim([min(rain_df.index), max(wb_591.index)])
ax1_1.legend([mpatches.Patch(color='#008000', alpha=0.5), mpatches.Patch(color='#FF0000', alpha=0.5),mpatches.Patch(color='#0000FF', alpha=0.7)], label_list).draggable()
# ax1_1.legend([mpatches.Patch(color='#0000FF', alpha=0.5), mpatches.Patch(color='#FF0000', alpha=0.5)], label_list).draggable()
ax1_1.set_ylabel(r"Flow ($m^{3} \, day^{-1}$)")
ax1.set_ylabel(r"Rainfall ($mm \, hr^{-1}$)")
# ax1_1.set_title("Check dam 591")
#
# fig.autofmt_xdate()
# # print evap
plt.show()
# raise SystemExit(0)
"""
#line plot
fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, facecolor='white')
bar_rain = ax1.bar(rain_df.index, rain_df['rain (mm)'], width=[(rain_df.index[j+1]-rain_df.index[j]).days for j in range(len(rain_df.index)-1)] + [30], color='#0000FF',alpha=0.9,label= 'Rainfall (mm)')
ax1.set_ylim([0, 300])
ax1.invert_yaxis()
for t1 in ax1.get_yticklabels():
    t1.set_color(dark_2_colors[2])
ax1_1 = ax1.twinx()
label_list = ['Overflow', 'Inflow', "Rainfall"]
stack = ax1_1.stackplot(wb_591.index, wb_591['overflow(cu.m)'], wb_591['Inflow (cu.m)'], colors=['#008000', '#FF0000'], alpha=0.5)
# display only 3 ticks
locator_1 = MaxNLocator(3)
locator_1_1 = MaxNLocator(3)
month_locator = mdates.MonthLocator(interval=3)
xfmt = mdates.DateFormatter('%b-%Y')
ax1.yaxis.set_major_locator(locator_1)
ax1_1.yaxis.set_major_locator(locator_1_1)
ax1_1.xaxis.set_major_locator(month_locator)
ax1_1.xaxis.set_major_formatter(xfmt)
ax1_1.set_xlim([min(rain_df.index), max(wb_591.index)])
ax1_1.legend([mpatches.Patch(color='#008000', alpha=0.5), mpatches.Patch(color='#FF0000', alpha=0.5),mpatches.Patch(color='#0000FF', alpha=0.9)], label_list)
# ax1_1.legend([mpatches.Patch(color='#0000FF', alpha=0.5), mpatches.Patch(color='#FF0000', alpha=0.5)], label_list).draggable()
ax1_1.set_ylabel(r"Flow ($m^{3} \, day^{-1}$)")
ax1.set_ylabel(r"Rainfall ($mm \, day^{-1}$)")
# ax1_1.set_title("Check dam 591")
#
fig.autofmt_xdate(rotation=90)
# # print evap
plt.show()
raise SystemExit(0)

#flow duration curve
inflow = wb_591['Inflow (cu.m)'].values
outflow = wb_591['overflow(cu.m)'].values
mean_inflow = np.mean(inflow)
mean_outflow = np.mean(outflow)
sigma_inflow = np.std(inflow)
sigma_outflow = np.std(outflow)
inflow_fit = stats.norm.pdf(sorted(inflow), mean_inflow, sigma_inflow)
outflow_fit = stats.norm.pdf(sorted(outflow), mean_outflow, sigma_outflow)
fig = plt.figure()
plt.plot(inflow_fit, inflow, '-ro', label='Inflow')
plt.plot(outflow_fit, outflow, '-go', label="Outflow")
plt.xlim([-0.0020, 0.0040])
plt.legend().draggable()
plt.show()
"""
# barplot
fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, facecolor='white')
bar_rain = ax1.bar(max_daily_rain_intensity_mm_hr.index, max_daily_rain_intensity_mm_hr['rain (mm)'], width=[(max_daily_rain_intensity_mm_hr.index[j+1]-max_daily_rain_intensity_mm_hr.index[j]).days for j in range(len(max_daily_rain_intensity_mm_hr.index)-1)] + [30], color='#000000',alpha=0.5,label = 'Rainfall (mm)')
ax1.set_ylim([0, 150])
ax1.invert_yaxis()
for t1 in ax1.get_yticklabels():
    t1.set_color(dark_2_colors[2])
ax1_1 = ax1.twinx()
label_list = [ 'Inflow','Overflow', "Rainfall"]
outflow = ax1_1.bar(wb_591.index, wb_591['Inflow (cu.m)'], color='#FF0000', alpha=0.5, label="Inflow",width=[(wb_591.index[j+1]-wb_591.index[j]).days for j in range(len(wb_591.index)-1)] + [30])
inflow = ax1_1.bar(wb_591.index, wb_591['overflow(cu.m)'], color='#0000FF', alpha=0.5, label="Outflow",width=[(wb_591.index[j+1]-wb_591.index[j]).days for j in range(len(wb_591.index)-1)] + [30])
# display only 3 ticks
locator_1 = MaxNLocator(3)
locator_1_1 = MaxNLocator(3)
month_locator = mdates.MonthLocator(interval=3)
xfmt = mdates.DateFormatter('%b-%Y')
# ax1.yaxis.set_major_locator(locator_1)
# ax1_1.yaxis.set_major_locator(locator_1_1)
# ax1_1.xaxis.set_major_locator(month_locator)
# ax1_1.xaxis.set_major_formatter(xfmt)
ax1_1.set_xlim([min(rain_df.index), max(wb_591.index)])
ax1_1.legend([mpatches.Patch(color='#FF0000', alpha=0.5), mpatches.Patch(color='#0000FF', alpha=0.5),mpatches.Patch(color='#000000', alpha=0.5)], label_list).draggable()
# ax1_1.legend([mpatches.Patch(color='#0000FF', alpha=0.5), mpatches.Patch(color='#FF0000', alpha=0.5)], label_list).draggable()
ax1_1.set_ylabel(r"Flow ($m^{3} \, day^{-1} $)")
ax1.set_ylabel(r"Rainfall ($mm \, day^{-1}$)")
# ax1_1.set_title("Check dam 591")
#
fig.autofmt_xdate(rotation=90)
# # print evap
plt.show()
raise SystemExit(0)
# bar plot

# print wb_591.head()
# latexify(fig_width=15, fig_height=10)
# fig, ax1 = plt.subplots(nrows=1,ncols=1, sharex=True, facecolor='white')
# line_1, = ax1.plot(stage_591_df.index, stage_591_df['stage(m)'], color='#a70c0b',linestyle='-', lw=3, alpha=0.75)
# ax1_1 = ax1.twinx()
# ax1_2 = ax1.twinx()
# bar_1_1 = ax1_1.bar(rain_df.index, rain_df['rain (mm)'], 0.85, color='#7570b3',alpha=0.85, label = 'Rainfall (mm)')
# ax1_1.invert_yaxis()
# bar_1 = ax1_2.bar(wb_591.index, wb_591['infiltration(cu.m)'], 1.15, color='#d95f02',alpha=0.85,label=r"\textbf{Infiltration ($m^3/day$}")
# bar_1_2 = ax1_2.bar(wb_591.index, wb_591['Evaporation (cu.m)'], 1.15, color='#1b9e77',alpha=0.85, label=r"\textbf{Evaporation ($m^3/day$)}")
# # bar_1_3 = ax1_2.bar(wb_591.index, wb_591['overflow(cu.m)'], 1.15, color='#66a61e', alpha=0.85, label=r"\textbf{Overflow ($m^3/day)}")
# for t1 in ax1_2.get_yticklabels():
#     t1.set_color('#d95f02')
# ax1_2.yaxis.label.set_color('#d95f02')
#
# lns = [bar_1_1, bar_1, bar_1_2, line_1]
# labs = [r'\textbf{Rainfall ($mm$)}', r"\textbf{Infiltration ($m^3/day$)}", r"\textbf{Evaporation ($m^3/day$)}", r"\textbf{Stage ($m$)}"]
# ax1.legend(lns, labs, loc='upper center', fancybox=True, ncol=4, bbox_to_anchor=(0.5, -0.05),prop={'size':30} ).draggable()
# ax1.set_title("Check Dam 591")
# ax1.yaxis.set_label_position('left')
# ax1.yaxis.set_ticks_position('left')
# for t1 in ax1.get_yticklabels():#a70c0b
#     t1.set_color('#d95f02')
# ax1.set_axis_bgcolor('white')
# for t1 in ax1_2.get_yticklabels():
#     t1.set_color('#203a72')
# locator_1 = MaxNLocator(3)
# locator_2 = MaxNLocator(3)
# locator_1_1 = MaxNLocator(3)
# locator_1_2 = MaxNLocator(3)
# locator_2_1 = MaxNLocator(3)
# locator_2_2 = MaxNLocator(3)
# ax1.yaxis.set_major_locator(locator_1)
# # ax2.yaxis.set_major_locator(locator_2)
# # ax1_1.yaxis.set_major_locator(locator_1_1)
# ax1_1.yaxis.set_major_locator(locator_1_1)
# ax1_2.yaxis.set_major_locator(locator_1_2)
# # ax2_2.yaxis.set_major_locator(locator_2_2)
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.spines['bottom'].set_visible(False)
# ax1.spines['left'].set_position(('outward', 50))
# ax1.yaxis.set_ticks_position('left')
# ax1.xaxis.set_ticks_position('bottom')
# ax1.tick_params(axis='y', colors='#a70c0b')
# ax1.spines['left'].set_color('#a70c0b')
# # ax1_2.set_ylabel("Stage (m)")
# ax1.yaxis.label.set_color('#a70c0b')
# # ax1.set_ylabel('Rainfall (mm)')
# ax1_1.yaxis.set_label_position('left')
# ax1_1.yaxis.set_ticks_position('left')
# ax1_1.yaxis.label.set_color('#7570b3')
# ax1_2.yaxis.label.set_color('#1b9e77')
# plt.tight_layout()
# fig.autofmt_xdate(rotation=90)
# # plt.savefig('/media/kiruba/New Volume/AGU/poster/agu_checkdam/image/evap_infilt.pdf', dpi=400)
# plt.show()



