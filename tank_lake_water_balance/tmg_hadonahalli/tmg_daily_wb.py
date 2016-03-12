__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import checkdam.checkdam as cd
from datetime import timedelta
import ccy_classic_lstsqr
from bokeh.plotting import figure, show, output_file
from bokeh.models import LinearAxis, Range1d
from bokeh.charts import Bar, output_file, show


"""
Variables
"""
full_stage = 3.8  # meter
date_format = '%Y-%m-%d %H:%M:%S'
daily_format = '%Y-%m-%d'
stage_cutoff = 0.1

# tank water level


water_level_file = '/media/kiruba/New Volume/milli_watershed/cumulative impacts/tmg_lake/stage_tmg.csv'
water_level = pd.read_csv(water_level_file, header=0, sep=',')
water_level['date_time'] = pd.to_datetime(water_level['date_time'], format=date_format)
water_level.set_index(water_level['date_time'], inplace=True)
water_level.drop('date_time', 1, inplace=True)
print water_level.head()
print water_level.tail()
# rain_file
rain_file = '/media/kiruba/New Volume/milli_watershed/cumulative impacts/had_rainfall.csv'
rain_df = pd.read_csv(rain_file, sep=',', header=0)
# set index
rain_df['Date_Time'] = pd.to_datetime(rain_df['date_time'], format=date_format)
rain_df.set_index(rain_df['Date_Time'], inplace=True)
# sort based on index
rain_df.sort_index(inplace=True)
# drop date time column
rain_df = rain_df.drop('Date_Time', 1)
rain_df['index'] = rain_df.index
rain_df.drop_duplicates(subset='index', keep='last', inplace=True)
del rain_df['index']
rain_df.sort_index(inplace=True)
rain_df = rain_df[min(water_level.index):]
print rain_df.head()
print rain_df.tail()


# p = figure(x_axis_type='datetime', title="TMG Lake level", y_range=(0, max(water_level['stage(m)'])))
# p.line(water_level.index, water_level['stage(m)'], color='navy', alpha=0.5)
# p.extra_y_ranges['rain'] = Range1d(max(rain_df['diff']), 0)
# p.rect(x= rain_df.index, y=rain_df['diff']/2, height=rain_df['diff'], width=1, color="red", y_range_name="rain")
# p.add_layout(LinearAxis(y_range_name="rain"), 'right')
# output_file('/media/kiruba/New Volume/milli_watershed/cumulative impacts/tmg_lake/stage.html')
# show(p)


