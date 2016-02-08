__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools


def load_decagon_sm_data_as_df(csv_file, no_of_sensors=4, date_time_format='%m/%d/%Y %I:%M %p'):
    names = ['date_time']
    for i in xrange(1, no_of_sensors+2):
        names.append(('port_{0:d}'.format(i)))
    df = pd.read_csv(csv_file, skiprows=3, sep=',', names=names, usecols=xrange(no_of_sensors+1))
    df['date_time'] = pd.to_datetime(df['date_time'], format=date_time_format)
    df.set_index(df['date_time'], inplace=True)
    df.drop(['date_time'], inplace=True, axis=1)
    return df

# rainfall dataset
rain_file = '/media/kiruba/New Volume/milli_watershed/cumulative impacts/had_rainfall.csv'
rain_df = pd.read_csv(rain_file, sep=',')
rain_df['date_time'] = pd.to_datetime(rain_df['date_time'], format='%Y-%m-%d %H:%M:%S')
rain_df.set_index(rain_df['date_time'], inplace=True)
rain_df.drop(['date_time'], inplace=True, axis=1)
print rain_df.head()

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
fig, ax_1 = plt.subplots(nrows=1, ncols=1, sharex=True, facecolor='white')
bar_1 = ax_1.bar(rain_df.index, rain_df['diff'], width=15.0/(len(rain_df.index) - 1), color='#203a72', alpha=0.85, label='Rainfall (mm)')
ax_1.invert_yaxis()
for t1 in ax_1.get_yticklabels():
    t1.set_color('#203a72')
ax_1_1 = ax_1.twinx()
# ax_1_1.plot(eu_dec_sm_df.index, eu_dec_sm_df['port_1'], 'r', label='30 cm')
ax_1_1.plot(eu_dec_sm_df.index, eu_dec_sm_df['port_2'], 'b', label='1 m')
# ax_1_1.plot(eu_dec_sm_df.index, eu_dec_sm_df['port_3'], 'g', label='2 m')
# ax_1_1.plot(eu_dec_sm_df.index, eu_dec_sm_df['port_4'], 'm', label='3 m')
ax_1_1.legend().draggable()
plt.show()

