__author__ = 'kiruba'
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import itertools
from PyQt4 import QtCore, QtGui
import sys

# gps  waypoint along with boundary points file in csv, with way point number, lat, long
gps_waypoint_file = '/media/kiruba/New Volume/milli_watershed/smg_lake_bathymetry/process/gps_waypoint.csv'
#  read the csv file and convert it to pandas dataframe
gps_waypoint_df = pd.read_csv(gps_waypoint_file)
#  select few columns that we are interested in
gps_waypoint_df = gps_waypoint_df[['ele,N,24,15', 'time,D', 'name,C,254', 'latitude,N,21,6', 'longitude,N,21,6']]
#  survey file has way point no and depth, time
survey_file = '/media/kiruba/New Volume/milli_watershed/smg_lake_bathymetry/process/survey_data.csv'
#  read the csv file and convert it to pandas dataframe
survey_df = pd.read_csv(survey_file)
#  select few columns that we are interested in
survey_df = survey_df[['r/gps_waypoint', 'r/depth_ft']]
# rename the columns, remove r/gps_waypoint
survey_df.rename(columns=lambda x: x[2:], inplace=True)
print survey_df.head()
gps_waypoint_df.columns.values[2] = 'gps_waypoint'
print gps_waypoint_df.head()
# merge two dataframes based on column gps_waypoint, keep the gps_waypoint df keys
merged_df = pd.merge(survey_df, gps_waypoint_df, on='gps_waypoint', how='right')
# assign zero depth to boundary
merged_df.loc[73:, ['gps_waypoint', 'depth_ft']] = 0.0
merged_df.loc[:, 'depth_ft'] = -merged_df['depth_ft']
print merged_df.head()
# save the merged file as csv
merged_df.to_csv('/media/kiruba/New Volume/milli_watershed/smg_lake_bathymetry/process/smg_bathymetry.csv')


