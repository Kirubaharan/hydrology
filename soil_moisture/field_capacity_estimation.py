__author__ = 'kiruba'
import matplotlib
matplotlib.use('Qt4Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import matplotlib.rcsetup as rcsetup
print(rcsetup.all_backends)
# from numpy import *
# import Gnuplot, Gnuplot.funcutils
from bokeh.plotting import figure, output_file, show
from bokeh.plotting import *
from bokeh.models import HoverTool
from bokeh.models.ranges import Range1d
from bokeh.models import LinearAxis

date_time_format = '%Y-%m-%d %H:%M:%S'

def load_decagon_sm_data_as_df(csv_file, no_of_sensors=4, date_time_format='%m/%d/%Y %I:%M %p'):
    names = ['date_time']
    for i in xrange(1, no_of_sensors+2):
        names.append(('port_{0:d}'.format(i)))
    df = pd.read_csv(csv_file, skiprows=3, sep=',', names=names, usecols=xrange(no_of_sensors+1))
    df['date_time'] = pd.to_datetime(df['date_time'], format=date_time_format)
    df.set_index(df['date_time'], inplace=True)
    df.drop(['date_time'], inplace=True, axis=1)
    return df

def plot_decagon_sm_df(sm_df, rain_df, ports=[1, 2, 3, 4]):
    rain_df = rain_df[min(sm_df.index).strftime(date_time_format):max(sm_df.index).strftime(date_time_format)]
    fig, ax_1 = plt.subplots(nrows=1, ncols=1, sharex=True, facecolor='white')
    bar_1 = ax_1.bar(rain_df.index, rain_df['diff'], width=1, color='#203a72', alpha=0.85, label='Rainfall (mm)')
    ax_1.xaxis_date()
    ax_1.invert_yaxis()
    for t1 in ax_1.get_yticklabels():
         t1.set_color('#203a72')
    ax_1_1 = ax_1.twinx()
    for port in ports:
        ax_1_1.plot(sm_df.index,sm_df[('port_{0}'.format(port))], label='port_{0}'.format(port))
    ax_1_1.xaxis_date()
    ax_1_1.legend()
    plt.show()



# rainfall dataset
rain_file = '/media/kiruba/New Volume/milli_watershed/cumulative impacts/had_rainfall.csv'
rain_df = pd.read_csv(rain_file, sep=',')
rain_df['date_time'] = pd.to_datetime(rain_df['date_time'], format='%Y-%m-%d %H:%M:%S')
rain_df.set_index(rain_df['date_time'], inplace=True)
rain_df.drop(['date_time'], inplace=True, axis=1)
print rain_df.head()
# resample to 30 min
# rain_df = rain_df.resample('30Min', how=np.sum, label='right', closed='right')
rain_df = rain_df.resample('D', how=np.sum, label='right', closed='right')
"""
# eucalyptus_decagon
eu_dec_block_1_file = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Eucalyptus site/Eucalyptus_Decagon sensors/EM28189 30Jul14-1132.csv'
eu_dec_block_1 = load_decagon_sm_data_as_df(eu_dec_block_1_file)
eu_dec_block_2_file = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Eucalyptus site/Eucalyptus_Decagon sensors/EM28189 13Oct14-1221.csv'
eu_dec_block_2 = load_decagon_sm_data_as_df(eu_dec_block_2_file)
eu_dec_block_3_file = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Eucalyptus site/Eucalyptus_Decagon sensors/EM28189 8May15-1509.csv'
eu_dec_block_3 = load_decagon_sm_data_as_df(eu_dec_block_3_file, date_time_format = '%d-%b-%y %I:%M %p')
eu_dec_block_4_file = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Eucalyptus site/Eucalyptus_Decagon sensors/EM28189 8Dec15-1543.csv'
eu_dec_block_4 = load_decagon_sm_data_as_df(eu_dec_block_4_file)
eu_dec_block_5_file = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Eucalyptus site/Eucalyptus_Decagon sensors/EM28189 6Mar15-1243.csv'
eu_dec_block_5 = load_decagon_sm_data_as_df(eu_dec_block_5_file)
# print eu_dec_block_1.head()
eu_dec_sm_df = pd.concat([eu_dec_block_1, eu_dec_block_2, eu_dec_block_3, eu_dec_block_4, eu_dec_block_5], axis=0)
eu_dec_sm_df.sort_index(inplace=True)
print eu_dec_sm_df.head()
rain_df = rain_df[min(eu_dec_sm_df.index).strftime('%Y-%m-%d %H:%M:%S'):max(eu_dec_sm_df.index).strftime('%Y-%m-%d %H:%M:%S')]
plot_decagon_sm_df(sm_df=eu_dec_sm_df, rain_df=rain_df)

raise SystemExit(0)
"""
"""
# rainfed decagon
rf_dec_block_1_file = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Rainfed site/EM28188 6Mar15-1203.csv'
rf_dec_block_1 = load_decagon_sm_data_as_df(rf_dec_block_1_file)
rf_dec_block_2_file = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Rainfed site/EM28188 10Dec15-1422.csv'
rf_dec_block_2 = load_decagon_sm_data_as_df(rf_dec_block_2_file)
rf_dec_block_3_file = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Rainfed site/EM28188 14Jan15-1309.csv'
rf_dec_block_3 = load_decagon_sm_data_as_df(rf_dec_block_3_file)
rf_dec_block_4_file = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Rainfed site/EM28188 28Jan16-1216_28_01_2016.csv'
rf_dec_block_4 = load_decagon_sm_data_as_df(rf_dec_block_4_file)
rf_dec_block_5_file = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Rainfed site/EM28188 30Jul14-1303.csv'
rf_dec_block_5 = load_decagon_sm_data_as_df(rf_dec_block_5_file)

rf_dec_sm_df = pd.concat([rf_dec_block_1, rf_dec_block_2, rf_dec_block_3, rf_dec_block_4, rf_dec_block_5], axis=0)
rf_dec_sm_df.sort_index(inplace=True)
print rf_dec_sm_df.head()
rain_df = rain_df[min(rf_dec_sm_df.index).strftime('%Y-%m-%d %H:%M:%S'):max(rf_dec_sm_df.index).strftime('%Y-%m-%d %H:%M:%S')]
plot_decagon_sm_df(sm_df=rf_dec_sm_df, rain_df=rain_df)
raise SystemExit(0)
"""
# DRIP grapes decagon
file_drip_dec_block_1 = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Drip irrigation site/EM28190 10Dec15-1449.csv'
drip_dec_block_1 = load_decagon_sm_data_as_df(file_drip_dec_block_1)
file_drip_dec_block_2 = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Drip irrigation site/EM28190 13Mar15-1429.csv'
drip_dec_block_2 = load_decagon_sm_data_as_df(file_drip_dec_block_2)
file_drip_dec_block_3 = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Drip irrigation site/EM28190 28Jan16-1243_28_01_2016.csv'
drip_dec_block_3 = load_decagon_sm_data_as_df(file_drip_dec_block_3)
file_drip_dec_block_4 = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Drip irrigation site/EM28190 30Jul14-1331.csv'
drip_dec_block_4 = load_decagon_sm_data_as_df(file_drip_dec_block_4)

drip_dec_sm_df = pd.concat([drip_dec_block_1, drip_dec_block_2, drip_dec_block_3, drip_dec_block_4], axis=0)
drip_dec_sm_df.sort_index(inplace=True)
drip_dec_sm_df = drip_dec_sm_df['2014-06-19':]
print drip_dec_sm_df.head()
drip_dec_sm_df.to_csv('/media/kiruba/New Volume/milli_watershed/soil_moisture/Drip irrigation site/drip_dec_sm_df.csv')
rain_df = rain_df[min(drip_dec_sm_df.index).strftime(date_time_format):max(drip_dec_sm_df.index).strftime(date_time_format)]
plot_decagon_sm_df(drip_dec_sm_df, rain_df)
raise SystemExit(0)
"""
output_file('/media/kiruba/New Volume/milli_watershed/soil_moisture/Drip irrigation site/drip_dec.html', title="Drip Irrigation Soil Moisture")
TOOLS="resize,crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"
# source = ColumnDataSource(data=dict(time=drip_dec_sm_df.index.strftime('%d-%m-%Y')))
p_1 = figure(tools=TOOLS,width=600, height= 400, x_axis_type = 'datetime', title='port_1')
p_1.line(drip_dec_sm_df.index, drip_dec_sm_df['port_1'], color='navy', alpha=0.5)
# p_1.select(dict(type=HoverTool)).tooltips = {"Date":"$index", "VWC":"$y"}
p_1.grid.grid_line_alpha=0
p_1.xaxis.axis_label = 'Date'
p_1.yaxis.axis_label = 'VWC'
p_2 = figure(tools=TOOLS,width=600, height= 400, x_axis_type = 'datetime', x_range=p_1.x_range, y_range=p_1.y_range, title='port_2')
p_2.line(drip_dec_sm_df.index, drip_dec_sm_df['port_2'], color='firebrick', alpha=0.5)
# p_2.select(dict(type=HoverTool)).tooltips = {"Date":"$x", "VWC":"$y"}
p_2.grid.grid_line_alpha=0
p_3 = figure(tools=TOOLS, width=600, height= 400, x_axis_type = 'datetime',x_range=p_1.x_range, y_range=p_1.y_range, title='port_3')
p_3.line(drip_dec_sm_df.index, drip_dec_sm_df['port_3'], color='aqua', alpha=0.5)
# p_3.select(dict(type=HoverTool)).tooltips = {"Date":"$x", "VWC":"$y"}
p_3.grid.grid_line_alpha=0
p_4 = figure(tools=TOOLS, width=600, height= 400, x_axis_type = 'datetime',x_range=p_1.x_range, y_range=p_1.y_range, title='port_4')
p_4.line(drip_dec_sm_df.index, drip_dec_sm_df['port_4'], color='olive', alpha=0.5)
# p_4.select(dict(type=HoverTool)).tooltips = {"Date":"$x", "VWC":"$y"}
p_4.grid.grid_line_alpha=0
p = gridplot([[p_1, p_2],[p_3, p_4]])
show(p)
"""
"""
# Cabbage - Flood Irrigation
file_fi_dec_block_1 = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Flood irrigation site/EM28187 10Dec15-1514.csv'
file_fi_dec_block_2 = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Flood irrigation site/EM28187 10Oct14-1232.csv'
file_fi_dec_block_3 = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Flood irrigation site/EM28187 28Jan16-1318_28_01_2016.csv'
file_fi_dec_block_4 = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Flood irrigation site/EM28187 30Jul14-1354.csv'

fi_dec_block_1 = load_decagon_sm_data_as_df(file_fi_dec_block_1)
fi_dec_block_2 = load_decagon_sm_data_as_df(file_fi_dec_block_2)
fi_dec_block_3 = load_decagon_sm_data_as_df(file_fi_dec_block_3)
fi_dec_block_4 = load_decagon_sm_data_as_df(file_fi_dec_block_4)

fi_dec_sm_df = pd.concat([fi_dec_block_1, fi_dec_block_2, fi_dec_block_3, fi_dec_block_4], axis=0)
fi_dec_sm_df.sort_index(inplace=True)

output_file('/media/kiruba/New Volume/milli_watershed/soil_moisture/Flood irrigation site/fi_dec.html', title="Flood Irrigation Soil Moisture")
TOOLS="resize,crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"
# source = ColumnDataSource(data=dict(time=drip_dec_sm_df.index.strftime('%d-%m-%Y')))
p_1 = figure(tools=TOOLS,width=800, height= 600, x_axis_type = 'datetime', title='1 m')
p_1.line(fi_dec_sm_df.index, fi_dec_sm_df['port_1'], color='navy', alpha=0.5)
# p_1.select(dict(type=HoverTool)).tooltips = {"Date":"$index", "VWC":"$y"}
p_1.grid.grid_line_alpha=0
p_1.xaxis.axis_label = 'Date'
p_1.yaxis.axis_label = 'VWC'
p_2 = figure(tools=TOOLS,width=800, height= 600, x_axis_type = 'datetime', x_range=p_1.x_range, y_range=p_1.y_range, title='1m')
p_2.line(fi_dec_sm_df.index, fi_dec_sm_df['port_2'], color='firebrick', alpha=0.5)
# p_2.select(dict(type=HoverTool)).tooltips = {"Date":"$x", "VWC":"$y"}
p_2.grid.grid_line_alpha=0
p_3 = figure(tools=TOOLS, width=800, height= 600, x_axis_type = 'datetime',x_range=p_1.x_range, y_range=p_1.y_range, title='30 cm')
p_3.line(fi_dec_sm_df.index, fi_dec_sm_df['port_3'], color='aqua', alpha=0.5)
# p_3.select(dict(type=HoverTool)).tooltips = {"Date":"$x", "VWC":"$y"}
p_3.grid.grid_line_alpha=0
p_4 = figure(tools=TOOLS, width=800, height= 600, x_axis_type = 'datetime',x_range=p_1.x_range, y_range=p_1.y_range, title='30 cm')
p_4.line(fi_dec_sm_df.index, fi_dec_sm_df['port_4'], color='olive', alpha=0.5)
# p_4.select(dict(type=HoverTool)).tooltips = {"Date":"$x", "VWC":"$y"}
p_4.grid.grid_line_alpha=0
p = gridplot([[p_1, p_2],[p_3, p_4]])
show(p)
"""