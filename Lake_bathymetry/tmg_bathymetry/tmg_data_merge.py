__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools


gps_waypoint_file = '/media/kiruba/New Volume/milli_watershed/tmg_lake_bathymetry/process/gps_waypoint.csv'
gps_waypoint_df = pd.read_csv(gps_waypoint_file)
# print gps_waypoint_df.head()
# raise SystemExit(0)
gps_waypoint_df = gps_waypoint_df[['ele', 'time', 'gps_waypoi', 'latitude', 'longitude']]
# print gps_waypoint_df.head()
# raise SystemExit(0)
survey_file = '/media/kiruba/New Volume/milli_watershed/tmg_lake_bathymetry/process/tmg_bathymetry_2016_01_13_11_30_59.csv'
survey_df = pd.read_csv(survey_file)
# print survey_df.head()
# raise SystemExit(0)
# /media/kiruba/New Volume/milli_watershed/doddatumkur_lake_bathymetry/process/gps_waypoint_combined_utm.csv
survey_df = survey_df[['gps_repeat/time', 'gps_repeat/gps_waypoint', 'gps_repeat/depth_ft']]
survey_df.rename(columns=lambda x: x[11:], inplace=True)
# print survey_df.head()
gps_waypoint_df.columns.values[2] = 'gps_waypoint'
print gps_waypoint_df.head()
print(survey_df.head())
# raise SystemExit(0)
# merge dataframes
merged_df = pd.merge(survey_df, gps_waypoint_df, on='gps_waypoint', how='inner')
print merged_df.head()
# merged_df.loc[507:, ['time', 'time,D', 'gps_waypoint', 'depth_ft']] = 0.0
# merged_df.loc[:, 'depth_ft'] = -merged_df['depth_ft']
print merged_df.tail()
merged_df.to_csv('/media/kiruba/New Volume/milli_watershed/tmg_lake_bathymetry/process/tmg_bathymetry.csv')
