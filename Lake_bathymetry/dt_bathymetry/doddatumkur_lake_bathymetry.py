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
    stage_area_df.loc[:, 'volume_cu_m'] = 0.000
    volume_2 = 0.000
    for h1, h2 in cd.pairwise(stage_area_df.index):
        print h1, h2
        height_diff = abs(h1 - h2)
        area_1 = stage_area_df.loc[h1, 'Area_sq_m']
        area_2 = stage_area_df.loc[h2, 'Area_sq_m']
        volume_2 += conic_volume_estimate(area_1=area_1, area_2=area_2, height_diff=height_diff)
        stage_area_df.loc[h2, 'volume_cu_m'] = volume_2
    return stage_area_df

date_format = '%d/%m/%y %H:%M'

weather_file = '/media/kiruba/New Volume/milli_watershed/doddatumkur_lake_bathymetry/weather_1_8_2016_12_00_17.csv'
# convert to pandas dataframe
weather_df = pd.read_csv(weather_file, sep='\t', header=0)
#Drop seconds
weather_df['Time'] = weather_df['Time'].map(lambda x: x[ :5])
weather_df['Date_Time'] = pd.to_datetime(weather_df['Date'] + ' ' + weather_df['Time'], format=date_format)
weather_df.set_index(weather_df['Date_Time'], inplace=True)
weather_df.columns.values[6] = 'Air Temperature (C)'
weather_df.columns.values[7] = 'Min Air Temperature (C)'
weather_df.columns.values[8] = 'Max Air Temperature (C)'
weather_df.columns.values[15] = 'Canopy Temperature (C)'
weather_df['index'] = weather_df.index
weather_df.drop_duplicates(subset='index', take_last=True, inplace=True)
del weather_df['index']
weather_df = weather_df.sort()
# new_index = pd.date_range(start=min(weather_df.index), end=max(weather_df.index), freq='30min' )
# weather_df = weather_df.reindex(index=new_index, method=None)
# weather_df = weather_df.interpolate(method='time')
weather_df.index.name = "Date_Time"
# print weather_df.head()

# sort based on index
weather_df.sort_index(inplace=True)
# drop date time column
weather_df = weather_df.drop('Date_Time', 1)
# print weather_df.head()
# raise SystemExit(0)

dt_bathymetry_file = '/media/kiruba/New Volume/milli_watershed/doddatumkur_lake_bathymetry/process/dt_bathymetry_with_boundary.csv'
dt_bathymetry_df = pd.read_csv(dt_bathymetry_file)
# print(dt_bathymetry_df.head())
# raise SystemExit(0)
# select columns which we need and rename
dt_bathymetry_df = dt_bathymetry_df[[ 'time,D', 'time','gps_waypoi', 'depth_ft', 'latitude,N', 'longitude,']]
# print dt_bathymetry_df.head()
dt_bathymetry_df.columns.values[0] = 'date'
dt_bathymetry_df.columns.values[4] = 'latitude'
dt_bathymetry_df.columns.values[5] = 'longitude'

# drop seconds
dt_bathymetry_df['time'] = dt_bathymetry_df['time'].map(lambda x: x[ :5])
print dt_bathymetry_df[507: ]
# dummy date and time for boundary

# dt_bathymetry_df['Date_Time'] = pd.to_datetime(dt_bathymetry_df['date'] + " " + dt_bathymetry_df['time'], format="%m/%d/%y %H:%M")
# print dt_bathymetry_df.head()
# raise SystemExit(0)
# calibration
y_cal = np.array([100, 400, 800, 1200, 1600, 2000, 2500, 3000, 3500, 4000, 4500])
x_cal = np.array([1911, 2078, 2306, 2536, 2771, 3001, 3307, 3587, 3888, 4185, 4480])
a_stage = cd.polyfit(x_cal, y_cal, 1)
coeff_cal = a_stage['polynomial']
slope = coeff_cal[0]
interecept = coeff_cal[1]
# depth correction based on capacitance
block_1 = '/media/kiruba/New Volume/ACCUWA_Data/lake_water_level/2973/2973_009_001_08_12_2015.CSV'
block_1_df = cd.read_correct_ch_dam_data(block_1, calibration_slope=slope, calibration_intercept=interecept)


# print block_1_df.head()
# print block_1_df.tail()

bathymetry_stage_df = block_1_df["2015-12-02" : "2015-12-04"]
bathymetry_stage_df = bathymetry_stage_df.append(block_1_df['2015-11-19'])
bathymetry_stage_df.sort_index(inplace=True)
# print bathymetry_stage_df
bathymetry_rain_df = weather_df["2015-11-19" : "2015-12-04"]
# bathymetry_rain_df = bathymetry_rain_df.append(weather_df['2015-11-19'])
bathymetry_rain_df.sort_index(inplace=True)


fig = plt.figure()
plt.plot(bathymetry_stage_df.index, bathymetry_stage_df['stage(m)'], 'r-o')
plt.plot(bathymetry_rain_df.index, bathymetry_rain_df['Rain Collection (mm)'], 'b-o')
plt.show()

print(np.max(bathymetry_stage_df['stage(m)']))
print(np.min(bathymetry_stage_df['stage(m)']))
# print dt_bathymetry_df.head()
raise SystemExit(0)
# boundary
boundary_file = '/media/kiruba/New Volume/milli_watershed/doddatumkur_lake_bathymetry/process/dt_lake_boundary.csv'
boundary_df = pd.read_csv(boundary_file)
print boundary_df.head()
# 3d plot

X = dt_bathymetry_df['longitude']
Y = dt_bathymetry_df['latitude']
Z = dt_bathymetry_df['depth_ft']
print min(Z)

xi = np.linspace(X.min(), X.max(), 1000)
yi = np.linspace(Y.min(), Y.max(), 1000)
zi = griddata((X, Y), Z, (xi[None, :], yi[:, None]), method='linear', fill_value=1)    # create a uniform spaced grid
xig, yig = np.meshgrid(xi, yi)
# fig = plt.figure()
# plt.plot(dt_bathymetry_df['longitude'], dt_bathymetry_df['latitude'], 'ro')
# plt.show()
# raise SystemExit(0)

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.gca(projection='3d')
# ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xig, yig, zi, rstride=30, cstride=40, linewidth=0, cmap=cm.coolwarm)
boundary = plt.plot(boundary_df['longitude'], boundary_df['latitude'], 'r-o')
# points = plt.plot(dt_bathymetry_df['longitude'], dt_bathymetry_df['latitude'], 'ko')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

levels = [ -2, -2.5, -5, -7.5, -10, -12 ]
plt.figure()
boundary = plt.plot(boundary_df['longitude'], boundary_df['latitude'], 'r-o')
CS = plt.contourf(xi, yi, zi, len(levels), alpha=.75, cmap=cm.hot, levels=levels)
C = plt.contour(xi, yi, zi, len(levels), colors='black', linewidth=.5, levels=levels)
plt.clabel(C, inline=1, fontsize=10)
plt.colorbar(CS)
# plt.yticks(np.arange(0,100, 5))
# plt.xticks(np.arange(-30,25, 5))
plt.grid()
# plt.gca().invert_xaxis()
plt.show()
# area of lake boundary , stage = 0
p = boundary[0].get_path()
v = p.vertices
area_at_stage_zero = abs(poly_area(v))
print area_at_stage_zero
contour_area = negative_contour_area(CS)
# add boundary area as zero stage area
contour_area.append((0, area_at_stage_zero ))
print contour_area
cont_area_df = pd.DataFrame(sorted(contour_area), columns=['Z', 'Area_sq_m'])
print cont_area_df

fig = plt.figure()
plt.plot(cont_area_df['Z'], cont_area_df['Area_sq_m'], 'r-o')
plt.xlabel("Stage")
plt.ylabel("Area Sq.m")
plt.show()

cont_area_df.set_index(cont_area_df['Z'], inplace=True)
cont_area_df.sort_index(inplace=True)
print cont_area_df

stage_volume_df = estimate_stage_volume(cont_area_df)

print stage_volume_df

fig = plt.figure()
plt.plot(stage_volume_df['Z'], stage_volume_df['volume_cu_m'], 'g-o')
plt.show()

