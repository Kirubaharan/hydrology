__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
# from datetime import timedelta
from matplotlib.dates import date2num

daily_date_time_format = "%Y-%m-%d"
date_time_format = "%Y-%m-%d %H:%M:%S"
# as is model - check dams with actual storage capactity
file_as_is_model_output = '/media/kiruba/New Volume/milli_watershed/cumulative impacts/lake_inflow_df.csv'
as_is_model_output_df = pd.read_csv(file_as_is_model_output)
as_is_model_output_df['Date'] = pd.to_datetime(as_is_model_output_df['Date'], format=date_time_format)
as_is_model_output_df.set_index(as_is_model_output_df['Date'], inplace=True)
as_is_model_output_df.drop(['Date'], inplace=True, axis=1)
print as_is_model_output_df.head()
# no_checkdam_model_output - check dams with storage=0
file_no_checkdam_model_output = '/media/kiruba/New Volume/milli_watershed/cumulative impacts/lake_inflow_df_wo_checkdam.csv'
no_checkdam_model_output_df = pd.read_csv(file_no_checkdam_model_output)
no_checkdam_model_output_df['Date'] = pd.to_datetime(no_checkdam_model_output_df['Date'], format=date_time_format)
no_checkdam_model_output_df.set_index(no_checkdam_model_output_df['Date'], inplace=True)
no_checkdam_model_output_df.drop(['Date'], inplace=True, axis=1)
print no_checkdam_model_output_df.head()

rain_file = '/media/kiruba/New Volume/milli_watershed/cumulative impacts/had_rainfall.csv'
rain_df = pd.read_csv(rain_file, sep=',')
rain_df['date_time'] = pd.to_datetime(rain_df['date_time'], format='%Y-%m-%d %H:%M:%S')
rain_df.set_index(rain_df['date_time'], inplace=True)
rain_df.drop(['date_time'], inplace=True, axis=1)
print rain_df.head()

rain_df = rain_df.resample('D', how=np.sum, label='right', closed='right')
rain_df = rain_df[min(as_is_model_output_df.index).strftime(daily_date_time_format):max(as_is_model_output_df.index).strftime(daily_date_time_format)]
print rain_df.head()
print len(rain_df.index)
# http://stackoverflow.com/a/27998372/2632856
# https://plot.ly/matplotlib/bar-charts/
x = date2num(as_is_model_output_df.index.to_pydatetime())
fig, ax_1 = plt.subplots(nrows=1, ncols=1, sharex=True, facecolor='white')
rain = ax_1.bar(rain_df.index, rain_df['diff'], width= 1, color='blue', alpha=0.3, label='Rainfall (mm)')
ax_1.xaxis_date()
for t1 in ax_1.get_yticklabels():
    t1.set_color('blue')
ax_1.invert_yaxis()
ax_1_1 = ax_1.twinx()
lake_inflow_wo_cd = ax_1_1.bar(no_checkdam_model_output_df.index, no_checkdam_model_output_df['inflow_into_lake'], width=1, color='red', alpha=0.8, label="Lake Inflow without CD")
lake_inflow_actual = ax_1_1.bar(as_is_model_output_df.index, as_is_model_output_df['inflow_into_lake'], width=1, color='green', alpha=0.4, label='Lake Inflow with CD')
ax_1_1.xaxis_date()
# lake_inflow = ax_1_1.plot(as_is_model_output_df.index, as_is_model_output_df['inflow_into_lake'], 'g-o', label='Lake Inflow')
ax_1_1.set_ylabel('Rainfall (mm/day)')
ax_1.set_ylabel('Inflow (cu.m/day)')
ax_1_1.yaxis.set_ticks_position('left')
ax_1.yaxis.set_ticks_position('right')
ax_1.yaxis.labelpad=57
ax_1_1.yaxis.labelpad=57
ax_1_1.legend([rain, lake_inflow_actual, lake_inflow_wo_cd], ['Rainfall (mm/day)', 'Inflow - with CD (cu.m/day)', 'Inflow - without CD (cu.m/day)'], fancybox=True).draggable()
plt.title('Inflow into Thirumagondanahalli Lake (HAD watershed)')
fig.autofmt_xdate()
plt.show()

as_is_inflow = as_is_model_output_df['inflow_into_lake'].sum(axis=0)
no_check_dam_inflow = no_checkdam_model_output_df['inflow_into_lake'].sum(axis=0)

difference_inflow = no_check_dam_inflow - as_is_inflow

percentage_diff =  (difference_inflow/no_check_dam_inflow)*100.0

print "as is inflow = {0}".format(as_is_inflow)
print "no check dam inflow = {0}".format(no_check_dam_inflow)
print "difference in inflow = {0}".format(difference_inflow)
print "percentage  difference = {0}".format(percentage_diff)
