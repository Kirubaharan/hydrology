__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

# this file exported from qgis after doing identity overlay
#  qgis - processing- qgis geo algorithms - vector overlay tools - intersection
landuse_raw_file = '/media/kiruba/New Volume/milli_watershed/cumulative impacts/had_landuse_raw_file.csv'
landuse_df = pd.read_csv(landuse_raw_file, sep=',')
landuse_grouped = landuse_df.groupby(["check_dam", "Class"])['crop_Area']
# print landuse_grouped.first()
sum_landuse_grouped = landuse_grouped.sum()
print sum_landuse_grouped.head()
sum_landuse_grouped.to_csv('/media/kiruba/New Volume/milli_watershed/cumulative impacts/had_landuse_stats.csv')
# print landuse_df.head(5)
