__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools


def load_decagon_sm_data_as_df(csv_file, no_of_sensors=4):
    names = ['date_time']
    date_time_format = '%m/%d/%Y %I:%M %p'
    for i in xrange(1, no_of_sensors+2):
        names.append(('port_{0:d}'.format(i)))
    df = pd.read_csv(csv_file, skiprows=3, sep=',', names=names, usecols=xrange(no_of_sensors+1))
    df['date_time'] = pd.to_datetime(df['date_time'], format=date_time_format)
    df.set_index(df['date_time'], inplace=True)
    df.drop(['date_time'], inplace=True, axis=1)
    return df

# eucalyptus_decagon
eu_dec_block_1_file = '/media/kiruba/New Volume/milli_watershed/soil_moisture/Eucalyptus site/Eucalyptus_Decagon sensors/EM28189 30Jul14-1132.csv'
eu_dec_block_1 = load_decagon_sm_data_as_df(eu_dec_block_1_file)
print eu_dec_block_1.head()