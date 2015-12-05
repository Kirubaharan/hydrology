__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from mpl_toolkits.mplot3d import Axes3D

# 1
# 591 stage vs area
# read from csv file
cont_area_591_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/cont_area.csv'
cont_area_591_df = pd.read_csv(cont_area_591_file, sep=',', header=0, names=['sno', 'stage_m', 'total_area_sq_m'])
# drop serial no
cont_area_591_df.drop('sno', inplace=True, axis=1)
# convert - 0 to 0
cont_area_591_df.loc[-0.00, 'stage_m'] = 0.00
# set stage as index
cont_area_591_df.set_index(cont_area_591_df['stage_m'], inplace=True)
print cont_area_591_df
# 2
# 599 stage vs area
# read from csv file
cont_area_599_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_599/cont_area.csv'
cont_area_599_df = pd.read_csv(cont_area_599_file, sep=',', header=0, names=['sno', 'stage_m', 'total_area_sq_m'])
# drop serial no
cont_area_599_df.drop('sno', inplace=True, axis=1)
# convert - 0 to 0
cont_area_599_df.loc[-0.00, 'stage_m'] = 0.00
# set stage as index
cont_area_599_df.set_index(cont_area_599_df['stage_m'], inplace=True)

# 3
# 634 stage vs area
# read from csv file
cont_area_634_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/cont_area.csv'
cont_area_634_df = pd.read_csv(cont_area_634_file, sep=',', header=0, names=['sno', 'stage_m', 'total_area_sq_m'])
# drop serial no
cont_area_634_df.drop('sno', inplace=True, axis=1)
# convert - 0 to 0
cont_area_634_df.loc[-0.00, 'stage_m'] = 0.00
# set stage as index
cont_area_634_df.set_index(cont_area_634_df['stage_m'], inplace=True)

# 4
# 463 stage vs area
# read from csv file
cont_area_463_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/cont_area.csv'
cont_area_463_df = pd.read_csv(cont_area_463_file, sep=',', header=0, names=['sno', 'stage_m', 'total_area_sq_m'])
# drop serial no
cont_area_463_df.drop('sno', inplace=True, axis=1)
# convert - 0 to 0
cont_area_463_df.loc[-0.00, 'stage_m'] = 0.00
# set stage as index
cont_area_463_df.set_index(cont_area_463_df['stage_m'], inplace=True)

# 5
# 616 stage vs area
# read from csv file
cont_area_616_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_616/cont_area.csv'
cont_area_616_df = pd.read_csv(cont_area_616_file, sep=',', header=0, names=['sno', 'stage_m', 'total_area_sq_m'])
# drop serial no
cont_area_616_df.drop('sno', inplace=True, axis=1)
# convert - 0 to 0
cont_area_616_df.loc[-0.00, 'stage_m'] = 0.00
# set stage as index
cont_area_616_df.set_index(cont_area_616_df['stage_m'], inplace=True)


# 6
# 623 stage vs area
# read from csv file
cont_area_623_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_623/cont_area.csv'
cont_area_623_df = pd.read_csv(cont_area_623_file, sep=',', header=0, names=['sno', 'stage_m', 'total_area_sq_m'])
# drop serial no
cont_area_623_df.drop('sno', inplace=True, axis=1)
# convert - 0 to 0
cont_area_623_df.loc[-0.00, 'stage_m'] = 0.00
# set stage as index
cont_area_623_df.set_index(cont_area_623_df['stage_m'], inplace=True)

check_dam_list = [591, 599, 634, 463, 616, 623]

# for check_dam_no in check_dam_list:
#     exec("hist_{0}, xedges_{0}, yedges_{0} = np.histogram2d(cont_area_{0}_df['stage_m'], cont_area_{0}_df['total_area_cu_m'], bins=20)".format(check_dam_no))
# 
# fig = plt.figure()
# 
# for check_dam_no in check_dam_list:
#     print check_dam_no
#     exec("plt.pcolormesh(xedges_{0}, yedges_{0}, hist_{0})".format(check_dam_no))
# 
# plt.xlabel('stage')
# plt.ylabel('area_sq_m')
# plt.colorbar()
# plt.show()

# with 591
# stage_area_df = pd.concat((cont_area_591_df, cont_area_599_df, cont_area_634_df, cont_area_463_df, cont_area_616_df, cont_area_623_df), axis=0)
# print cont_area_df.head()
# without 591
stage_area_df = pd.concat((cont_area_599_df, cont_area_634_df, cont_area_463_df, cont_area_616_df, cont_area_623_df), axis=0)
area_fit = [25.40, 150.66, 204.46, 273.26, 327.95, 386.16, 446.14, 507.01, 581.10] #  274.14,
stage_fit = [0.07, 0.22, 0.39, 0.67, 0.83, 1.00, 1.13, 1.26, 1.43] # 0.53,
hist, xedges, yedges = np.histogram2d(stage_area_df['stage_m'], stage_area_df['total_area_sq_m'], bins=10)
# print hist
print xedges
print yedges
"""
collect points on click
http://stackoverflow.com/a/25525143/2632856
"""

coords = []

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print 'x = {0:0.2f}, y = {1:0.2f}'.format(ix, iy)
    global coords
    coords.append((ix, iy))
    return coords

hist_masked = np.ma.masked_where(hist==0, hist)
fig = plt.figure()
plt.pcolormesh(xedges, yedges, hist_masked)
plt.xlabel('stage')
plt.ylabel('area_sq_m')
plt.plot(stage_fit, area_fit, 'ro-')
plt.colorbar()
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
