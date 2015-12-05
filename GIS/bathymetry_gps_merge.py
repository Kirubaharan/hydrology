__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

input_folder = '/media/kiruba/New Volume/milli_watershed/doddatumkur_lake_bathymetry/process'
gps_way_point_file = input_folder + '/gps_waypoint_19_11_2015.csv'
gps_way_point_df = pd.read_csv(gps_way_point_file)
gps_way_point_df = gps_way_point_df[['ele,N,24,15', 'time,D', 'name,C,254', 'latitude,N,21,9', 'longitude,N,21,9']]
survey_file = input_folder + '/survey_data_19_11_2015.csv'
survey_df = pd.read_csv(survey_file)
survey_df = survey_df[['gps_repeat/time', 'gps_repeat/gps_waypoint', 'gps_repeat/depth_ft', 'gps_repeat/_gps_lat_long_latitude', 'gps_repeat/_gps_lat_long_longitude', 'gps_repeat/_gps_lat_long_altitude', 'gps_repeat/_gps_lat_long_precision']]
# rename columns, remove the string gps_repeat/
survey_df.rename(columns=lambda x: x[11:], inplace=True)
# create common name for waypoint
gps_way_point_df.columns.values[2] = 'gps_waypoint'
merged_df = pd.merge(survey_df, gps_way_point_df, on='gps_waypoint', how='inner')
# print survey_df.head()
# print gps_way_point_df.head()
print merged_df.head()
merged_df.to_csv(input_folder + '/19_11_15_dt_bathymetry.csv')