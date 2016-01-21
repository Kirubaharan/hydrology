__author__ = 'kiruba'
import pandas as pd
import matplotlib.pyplot as plt
import mpld3 as m
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import rc
from scipy.interpolate import griddata
import numpy as np
from matplotlib import cm
from matplotlib.path import *
from matplotlib.collections import PolyCollection
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
import checkdam.checkdam as cd
import checkdam.mynormalize as mn

def poly_area(xy):
    """
    Calculates polygon area
    x = xy[:,0], y[xy[:,1]
    :param xy:
    :return:
    """
    l = len(xy)
    s = 0.0
    for i in range(l):
        j = (i+1) % l
        s += (xy[j, 0] - xy[i, 0]) * (xy[j,1]+ xy[i,1])
    return -0.5*s



def negative_contour_area(mpl_obj):
    """
    Returns a array of contour levels and
    corresponding cumulative area of contours
    specifically used for calculating negative contour's area when the contours are depth of lake
    # Refer: Nikolai Shokhirev http://www.numericalexpert.com/blog/area_calculation/

    :param mpl_obj: Matplotlib contour object
    :return: [(level1, area1), (level1, area1+area2)]
    """
    n_c = len(mpl_obj.collections)  # n_c = no of contours
    print 'No. of contours = {0}'.format(n_c)
    # area = 0.0000
    cont_area_array = []
    for contour in range(n_c):
        n_p = len(mpl_obj.collections[contour].get_paths())
        zc = mpl_obj.levels[contour]
        print zc
        print n_p
        area = 0.000
        for path in range(n_p):
            p = mpl_obj.collections[contour].get_paths()[path]
            v = p.vertices
            l = len(v)
            s = 0.0000
            # plt.figure()
            # plt.fill(v[:, 0], v[:, 1], facecolor='b')
            # plt.grid()
            # plt.show()
            for i in range(l):
                j = (i + 1) % l
                s += (v[j, 0] - v[i, 0]) * (v[j, 1] + v[i, 1])
            poly_area = abs(0.5 * s)
            area += poly_area
        cont_area_array.append((zc, area))
    return cont_area_array

def conic_volume_estimate(area_1, area_2, height_diff):
    volume = (height_diff/3.0)*(area_1 + area_2 + (math.sqrt(area_1*area_2)))
    return volume

def estimate_stage_volume(stage_area_df):
    """
    Function to calculate stage volume from stage_area dataframe
    """
    # give default value as 0, this will get reset after each loop
    stage_area_df.loc[:, 'volume_cu_m'] = 0.000
    volume_2 = 0.000
    # loop, h -> (h0, h1), (h1, h2), (h2, h3)
    for h1, h2 in cd.pairwise(stage_area_df.index):
        # print h1, h2
        # estimate diff in height between two values
        height_diff = abs(h1 - h2)
        # find out corresponding area of h1, and h2
        area_1 = stage_area_df.loc[h1, 'Area_sq_m']
        area_2 = stage_area_df.loc[h2, 'Area_sq_m']
        # estimate volume using conic volume formula, see above function
        volume_2 += conic_volume_estimate(area_1=area_1, area_2=area_2, height_diff=height_diff)
        # assign volume to h2
        stage_area_df.loc[h2, 'volume_cu_m'] = volume_2
    return stage_area_df

# bathymetry csv file that has depth. lat, long
# modify as per your case
data_file = '/media/kiruba/New Volume/milli_watershed/smg_lake_bathymetry/process/smg_bathymetry_with_boundary.csv'
#  read the csv file and convert it to pandas dataframe
data_df = pd.read_csv(data_file)
#  select few columns that we are interested in
# modify as per your case
data_df = data_df[['depth_ft,N,24,15', 'latitude,N,N,24,15', 'longitude,,N,24,15']]
# rename columns
data_df.columns.values[:] = ['depth_ft', 'latitude', 'longitude']
# print first five rows of dataframe
print data_df.head()
# boundary csv file that has vertices of lake boundary with lat long
# modify as per your case
boundary_file = '/media/kiruba/New Volume/milli_watershed/smg_lake_bathymetry/process/smg_lake_boundary.csv'
#  read the csv file and convert it to pandas dataframe
boundary_df = pd.read_csv(boundary_file, sep='\t')
# print first five rows of dataframe
print boundary_df.head()
# assign variable names for columns for understanding
X = data_df['longitude']
Y = data_df['latitude']
Z = data_df['depth_ft']
print min(Z)

# creating uniform spaced grid
#  create 500 uniform points between smallest X value to largest, modify the number of points
xi = np.linspace(X.min(), X.max(), 500)
yi = np.linspace(Y.min(), Y.max(), 500)
# create uniform grid from data based on X, Y, Z, , do linear interpolation,
#  use fill value of 1 for areas outside the boundary/ where there is no data
zi = griddata((X, Y), Z, (xi[None, :], yi[:, None]), method='linear', fill_value=1)
# create meshgrid, necessary for contour
xig, yig = np.meshgrid(xi, yi)

"""
# contour
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xig, yig, zi, rstrisortde=30, cstride=40, linewidth=0, cmap=cm.coolwarm)
boundary = plt.plot(boundary_df['longitude'], boundary_df['latitude'], 'ro-')
fig.colorbar(surf, shrink= 0.5, aspect=5)
plt.show()
"""
# modify as per your case
# depths that needs to be contoured
levels = [-1, -2.0,  -2.5, -5, -7.5, -10, -12.4, -13.0]
# contour plot
fig = plt.figure(facecolor='white')
# CS = plt.contourf(xi, yi, zi, 10, cmap=cm.hot, origin='lower')
# C = plt.contour(xi, yi, zi, 10, colors='black', origin='lower')
boundary = plt.plot(boundary_df['longitude'], boundary_df['latitude'], 'ro-')
CS = plt.contourf(xi, yi, zi, len(levels), alpha=0.75, cmap=cm.hot, levels=levels)
C = plt.contour(xi, yi, zi, len(levels), colors='black', linewidth=0.5, levels=levels)
plt.clabel(C, inline=1, fontsize=10)
plt.colorbar(CS, shrink=0.5, aspect=5)
plt.grid()
plt.show()
# calculate area of lake boundary at stage = 0
# select the boundary object from plot
p = boundary[0].get_path()
# take vertices
v = p.vertices
# calculate area using poly_area function, see top
area_at_stage_zero = abs(poly_area(v))
print area_at_stage_zero
# calculate the stage area relationship using function negative contour area, see top for function detail
contour_area = negative_contour_area(CS)
# add boundary area as zero stage area
contour_area.append((0, area_at_stage_zero))
print contour_area
# create a pandas dataframe(stage area ) with column names
cont_area_df = pd.DataFrame(sorted(contour_area), columns=['Z', 'Area_sq_m'])
print cont_area_df
# save the csv file (stage area)
# modify as per your case
cont_area_df.to_csv('/media/kiruba/New Volume/milli_watershed/smg_lake_bathymetry/stage_area.csv')

# stage area plot
fig = plt.figure()
plt.plot(cont_area_df['Z'], cont_area_df['Area_sq_m'], 'r-o')
plt.xlabel("Stage")
plt.ylabel("Area Sq.m")
plt.show()

# stage volume estimation
# set depth as index for stage area dataframe
cont_area_df.set_index(cont_area_df['Z'], inplace=True)
# sort based on index
cont_area_df.sort_index(inplace=True)
print cont_area_df
# calculate stage volume  based on estimate_stage_volume function, see top for details
stage_volume_df = estimate_stage_volume(cont_area_df)

print stage_volume_df
# stage volume plot
fig = plt.figure()
plt.plot(stage_volume_df['Z'], stage_volume_df['volume_cu_m'], 'g-o')
plt.show()
