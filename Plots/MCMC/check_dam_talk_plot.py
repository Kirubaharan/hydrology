__author__ = 'kiruba'
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
import matplotlib
from matplotlib.ticker import MaxNLocator
# from statsmodels.nonparametric.smoothers_lowess import lowess
import brewer2mpl
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import  matplotlib.patches as mpatches
import scipy.stats as stats


# latex parameters
rc('font', **{'family': 'sans-serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=36)

dark_2_colors = brewer2mpl.get_map("Dark2", 'Qualitative', 7).mpl_colors


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
fig_1, ax = plt.subplots(1, figsize=(10, 5), facecolor='white')
bar_width = 1
bar_l = [i for i in range(len(results_pie_df['Percentage of E']))]
print bar_l
tick_pos = [i+(bar_width/2.0) for i in bar_l]
print tick_pos
totals = results_pie_df['Inflow (cu.m)']
evap = [i /j *100 for i, j in zip(results_pie_df['Evaporation (cu.m)'], totals)]
overflow = [i /j *100 for i, j in zip(results_pie_df['Overflow (cu.m)'], totals)]
infil = [i /j *100 for i, j in zip(results_pie_df['Infiltration (cu.m)'], totals)]
ax.bar(bar_l, evap, label='Evaporation', alpha=0.9, color='#019600', width=bar_width, edgecolor='white')
ax.bar(bar_l, overflow,bottom=evap, label='Overflow', alpha=0.9, color='#3C5F5A', width=bar_width, edgecolor='white')
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
raise SystemExit(0)

daily_format = '%Y-%m-%d'
datetime_format = '%Y-%m-%d %H:%M:%S'
# 591
file_591 = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/et_infilt_591_w_of.csv'
rain_a_file = '/media/kiruba/New Volume/ACCUWA_Data/weather_station/smgollahalli/ksndmc_rain.csv'
stage_591_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/stage_591.csv'
rain_df = pd.read_csv(rain_a_file, sep=',', header=0)
rain_df['Date_Time'] = pd.to_datetime(rain_df['Date_Time'], format=datetime_format)
rain_df.set_index(rain_df['Date_Time'], inplace=True)
# sort based on index
rain_df.sort_index(inplace=True)
# drop date time column
rain_df = rain_df.drop('Date_Time', 1)
rain_df = rain_df.resample('D', how=np.sum)
wb_591 = pd.read_csv(file_591, sep=',', header=0)
stage_591_df = pd.read_csv(stage_591_file, sep=',', header=0)
stage_591_df.set_index(pd.to_datetime(stage_591_df['Date'],format=datetime_format),  inplace=True)
wb_591.set_index(pd.to_datetime(wb_591['Date'], format=daily_format), inplace=True)
del wb_591['Date']
width = 300.0/len(wb_591.index)
# print 'width = %s'  %width
# wb_591 = wb_591[wb_591['Inflow (cu.m)'] > 0]
# wb_591 = wb_591[:'2014-12-31']
# rain_df = rain_df[:"2014-12-31"]
# print wb_591.head()
# stage_591_df = stage_591_df.resample('D', how=np.mean)
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
# line_stage, = ax1_1.plot(stage_591_df.index, stage_591_df['stage(m)'], color='#e41a1c',linestyle='-', lw=1.5, alpha=0.85)
# bar_overflow = ax1_3.bar(wb_591.index, wb_591['overflow(cu.m)'], width=[(wb_591.index[j+1]-wb_591.index[j]).days for j in range(len(wb_591.index)-1)] + [30], color=dark_2_colors[5], alpha=1)
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
# plt.show()
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
# plt.show()
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
# plt.show()
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
# plt.show()
# barplot
fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, facecolor='white')
bar_rain = ax1.bar(rain_df.index, rain_df['rain (mm)'], width=[(rain_df.index[j+1]-rain_df.index[j]).days for j in range(len(rain_df.index)-1)] + [30], color='#000000',alpha=0.5,label = 'Rainfall (mm)')
ax1.set_ylim([0, 300])
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
ax1.yaxis.set_major_locator(locator_1)
ax1_1.yaxis.set_major_locator(locator_1_1)
ax1_1.xaxis.set_major_locator(month_locator)
ax1_1.xaxis.set_major_formatter(xfmt)
ax1_1.set_xlim([min(rain_df.index), max(wb_591.index)])
ax1_1.legend([mpatches.Patch(color='#FF0000', alpha=0.5), mpatches.Patch(color='#0000FF', alpha=0.5),mpatches.Patch(color='#000000', alpha=0.5)], label_list).draggable()
# ax1_1.legend([mpatches.Patch(color='#0000FF', alpha=0.5), mpatches.Patch(color='#FF0000', alpha=0.5)], label_list).draggable()
ax1_1.set_ylabel(r"Flow ($m^{3} \, day^{-1} $)")
ax1.set_ylabel(r"Rainfall ($mm \, day^{-1}$)")
# ax1_1.set_title("Check dam 591")
#
fig.autofmt_xdate(rotation=90)
# # print evap
# plt.show()
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
pie1, text1, autotexts_1 = ax1.pie([evap_591, infilt_591, overflow_591], labels=['E', 'P', 'O'],
                                   colors=dark_2_colors[0:3],
                                   autopct='%i%%',
                                   )
legend = fig.legend([pie1[0], pie1[1], pie1[2]],["Evaporation", "Percolation", "Overflow"],fancybox=True, ncol=3,loc=(0.01,0.02)).draggable()
ax1.set_title("Check dam 591")
ax2 = plt.subplot(132,aspect=True,axisbg='w')
evap_599 = results_pie_df['Evaporation (cu.m)'][599]
infilt_599 = results_pie_df['Infiltration (cu.m)'][599]
overflow_599 = results_pie_df['Overflow (cu.m)'][599]
pie2, text2, autotexts_2 = ax2.pie([evap_599, infilt_599, overflow_599], labels=['E', 'P', 'O'],
                                   colors=dark_2_colors[0:3],
                                   autopct='%i%%')
ax2.set_title("Check dam 599")
ax3 = plt.subplot(133,aspect=True,axisbg='w')
evap_634 = results_pie_df['Evaporation (cu.m)'][634]
infilt_634 = results_pie_df['Infiltration (cu.m)'][634]
overflow_634 = results_pie_df['Overflow (cu.m)'][634]
pie3, text3, autotexts_3 = ax3.pie([evap_634, infilt_634, overflow_634], labels=['E', 'P', 'O'],
                                   colors=dark_2_colors[0:3],
                                   autopct='%i%%',
                                   )
ax3.set_title("Check dam 634")
for label in text1:
    label.set_fontsize(36)
for label in text2:
    label.set_fontsize(36)
for label in text3:
    label.set_fontsize(36)
plt.show()