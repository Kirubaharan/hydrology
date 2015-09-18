__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse


def crazy_histogram2d(x, y, bins=10):
    try:
        nx, ny = bins
    except TypeError:
        nx = ny = bins
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    dx = (xmax - xmin) / (nx - 1.0)
    dy = (ymax - ymin) / (ny - 1.0)

    weights = np.ones(x.size)

    # Basically, this is just doing what np.digitize does with one less copy
    xyi = np.vstack((x,y)).T
    xyi -= [xmin, ymin]
    xyi /= [dx, dy]
    xyi = np.floor(xyi, xyi).T

    # Now, we'll exploit a sparse coo_matrix to build the 2D histogram...
    grid = scipy.sparse.coo_matrix((weights, xyi), shape=(nx, ny)).toarray()

    return grid, np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)

# 1
# 591 stage vs volume
# read from csv file
stage_vol_591_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/stage_vol_new.csv'
stage_vol_591_df = pd.read_csv(stage_vol_591_file, sep=',', header=0, names=['sno', 'stage_m', 'total_vol_cu_m'])
# drop serial no
stage_vol_591_df.drop('sno', inplace=True, axis=1)
# convert - 0 to 0
stage_vol_591_df.loc[-0.00, 'stage_m'] = 0.00
# set stage as index
# stage_vol_591_df.set_index(stage_vol_591_df['stage_m'], inplace=True)

# 2
# 599 stage vs volume
# read from csv file
stage_vol_599_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_599/stage_vol.csv'
stage_vol_599_df = pd.read_csv(stage_vol_599_file, sep=',', header=0, names=['sno', 'stage_m', 'total_vol_cu_m'])
# drop serial no
stage_vol_599_df.drop('sno', inplace=True, axis=1)
# convert - 0 to 0
stage_vol_599_df.loc[-0.00, 'stage_m'] = 0.00
# set stage as index
# stage_vol_599_df.set_index(stage_vol_599_df['stage_m'], inplace=True)

# 3
# 634 stage vs volume
# read from csv file
stage_vol_634_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/stage_vol.csv'
stage_vol_634_df = pd.read_csv(stage_vol_634_file, sep=',', header=0, names=['sno', 'stage_m', 'total_vol_cu_m'])
# drop serial no
stage_vol_634_df.drop('sno', inplace=True, axis=1)
# convert - 0 to 0
stage_vol_634_df.loc[-0.00, 'stage_m'] = 0.00
# set stage as index
# stage_vol_634_df.set_index(stage_vol_634_df['stage_m'], inplace=True)

# 4
# 463 stage vs volume
# read from csv file
stage_vol_463_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/stage_vol.csv'
stage_vol_463_df = pd.read_csv(stage_vol_463_file, sep=',', header=0, names=['sno', 'stage_m', 'total_vol_cu_m'])
# drop serial no
stage_vol_463_df.drop('sno', inplace=True, axis=1)
# convert - 0 to 0
stage_vol_463_df.loc[-0.00, 'stage_m'] = 0.00
# set stage as index
# stage_vol_463_df.set_index(stage_vol_463_df['stage_m'], inplace=True)

# 5
# 616 stage vs volume
# read from csv file
stage_vol_616_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_616/stage_vol.csv'
stage_vol_616_df = pd.read_csv(stage_vol_616_file, sep=',', header=0, names=['sno', 'stage_m', 'total_vol_cu_m'])
# drop serial no
stage_vol_616_df.drop('sno', inplace=True, axis=1)
# convert - 0 to 0
stage_vol_616_df.loc[-0.00, 'stage_m'] = 0.00
# set stage as index
# stage_vol_616_df.set_index(stage_vol_616_df['stage_m'], inplace=True)


# 6
# 623 stage vs volume
# read from csv file
stage_vol_623_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_623/stage_vol.csv'
stage_vol_623_df = pd.read_csv(stage_vol_623_file, sep=',', header=0, names=['sno', 'stage_m', 'total_vol_cu_m'])
# drop serial no
stage_vol_623_df.drop('sno', inplace=True, axis=1)
# convert - 0 to 0
stage_vol_623_df.loc[-0.00, 'stage_m'] = 0.00
# set stage as index
# stage_vol_623_df.set_index(stage_vol_623_df['stage_m'], inplace=True)

check_dam_list = [591, 599, 634, 463, 616, 623]

# for check_dam_no in check_dam_list:
#     exec("hist_{0}, xedges_{0}, yedges_{0} = crazy_histogram2d(stage_vol_{0}_df['stage_m'], stage_vol_{0}_df['total_vol_cu_m'], bins=20)".format(check_dam_no))
#
# fig = plt.figure()
#
# for check_dam_no in check_dam_list:
#     print check_dam_no
#     exec("plt.pcolormesh(xedges_{0}, yedges_{0}, hist_{0})".format(check_dam_no))
#
# plt.xlabel('stage')
# plt.ylabel('volume_cu_m')
# plt.colorbar()
# plt.show()
# with 591
# stage_vol_df = pd.concat((stage_vol_591_df, stage_vol_599_df, stage_vol_634_df, stage_vol_463_df, stage_vol_616_df, stage_vol_623_df), axis=0)
# print stage_vol_df.head()
# without 591
stage_vol_df = pd.concat((stage_vol_599_df, stage_vol_634_df, stage_vol_463_df, stage_vol_616_df, stage_vol_623_df), axis=0)

hist, xedges, yedges = np.histogram2d(stage_vol_df['stage_m'], stage_vol_df['total_vol_cu_m'], bins=10)
hist_masked = np.ma.masked_where(hist==0, hist)
fig = plt.figure()
plt.pcolormesh(xedges, yedges, hist_masked)
plt.xlabel('stage')
plt.ylabel('volume_cu_m')
plt.colorbar()
plt.show()

