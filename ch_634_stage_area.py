__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from spread import  spread
from bisect import bisect_left, bisect_right
from matplotlib import rc
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.path import *
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib as mpl
import matplotlib.colors as mc

# latex parameters
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)

base_file = '/media/kiruba/New Volume/milli_watershed/stream_profile/634/base_profile_634.csv'
df_base = pd.read_csv(base_file, header=-1, skiprows=1)
# print df_base.head()
# slope_file = '/media/kiruba/New Volume/milli_watershed/stream_profile/616/slope_616.csv'
# df_slope = pd.read_csv(slope_file, header=0)
# print df_slope
df_base_trans = df_base.T
df_base_trans.columns = df_base_trans.ix[0, 0:]
# print df_base_trans
df_base_trans = df_base_trans.ix[1:, 1500:]
print df_base_trans
# raise SystemExit(0)

"""
Filling of profile
"""


def find_range(array, ab):
    if ab < max(array):
        start = bisect_left(array, ab)
        return array[start-1]
    else:
        return max(array)


def fill_profile(base_df, slope_df, midpoint_index):
    """

    :param base_df:  base profile
    :param slope_df: slope profile
    :param midpoint_index: index of midpoint(x=0)
    :return:
    """
    base_z = base_df.ix[midpoint_index, 0:]
    slope_z = slope_df.ix[ :, 1]
    base_y = base_z.index
    # print base_z
    # base_y_list =base_y.tolist()
    slope_y = slope_df.ix[:, 0]
    slope_z.index = slope_y
    # print slope_z.head()
        # print base_z
    new_base_df = base_df
    for y_s in slope_z.index:
        if y_s not in base_z.index.tolist():
            # print y_s
            y_t = find_range(base_y, y_s)
            template = base_df[y_t]
            z1 = template.ix[midpoint_index, ]
            # print z1
            z2 = slope_z[y_s]
            diff = z2 - z1
            # print template
            # print diff
            profile = template + diff
            profile.name = y_s
            # profile.loc[0] = y_s
            # profile = profile.sort_index()
            # print profile
            # no_of_col = len(base_df.columns)
            new_base_df = new_base_df.join(profile, how='right')
            # base_df.columns.values[no_of_col+1] = y_s
    return new_base_df


def set_column_sequence(dataframe, seq):
    '''Takes a dataframe and a subsequence of its columns, returns dataframe with seq as first columns'''
    cols = seq[:] # copy so we don't mutate seq
    for x in dataframe.columns:
        if x not in cols:
            cols.append(x)
    return dataframe[cols]

# created_profile = fill_profile(df_base_trans, df_slope, 7)

# created_profile = created_profile[sorted(created_profile.columns)]
created_profile = df_base_trans
# print created_profile.head()
sorted_df = created_profile.iloc[0:, 1:]
sorted_df = sorted_df[sorted(sorted_df.columns)]
sorted_df = sorted_df.join(created_profile.iloc[0:, 0], how='right')
created_profile = set_column_sequence(sorted_df, [1500])
# print created_profile.head()
# raise SystemExit(0)
"""
Create (x,y,z) point cloud
"""
z_array = created_profile.iloc[0:, 1:]
columns = z_array.columns
z_array = z_array.values
index = created_profile.iloc[0:,0]
df = pd.DataFrame(z_array, columns=columns).set_index(index)
data_1 = []
for y, row in df.iteritems():
    for x, z in row.iteritems():
        data_1.append((x, y, z))

data_1_df = pd.DataFrame(data_1, columns=['x', 'y', 'z'])
# print data_1_df.dtypes
# raise SystemExit(0)
X = data_1_df.x
Y = data_1_df.y
Z = data_1_df.z

## contour and 3d surface plotting
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.gca(projection='3d')
# ax = fig.add_subplot(1, 2, 1, projection='3d')
xi = np.linspace(X.min(), X.max(), 100)
yi = np.linspace(Y.min(), Y.max(), 100)
# print len(xi)
# print len(yi)
# print len(Z)
zi = griddata((X, Y), Z, (xi[None, :], yi[:, None]), method='linear')    # create a uniform spaced grid
xig, yig = np.meshgrid(xi, yi)
surf = ax.plot_wireframe(X=xig, Y=yig, Z=zi, rstride=5, cstride=3, linewidth=1)#, cmap=cm.coolwarm, antialiased=False)   # 3d plot
# inter_1 = []
# inter_1.append((xi, yi, zi))
# inter = pd.DataFrame(inter_1, columns=['x', 'y', 'z'])
# inter.to_csv('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/inter.csv')  # interpolation data output
# fig.colorbar(surf, shrink=0.5, aspect=5)
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.xlabel(r'\textbf{X} (m)')
# plt.ylabel(r'\textbf{Y} (m)')
# plt.title(r"Profile for 591", fontsize=16)
plt.gca().invert_xaxis()  # reverses x axis
# # ax = fig
# plt.savefig('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/linear_interpolation')
plt.show()
# raise SystemExit(0)
# ## trace contours
# Refer: Nikolai Shokhirev http://www.numericalexpert.com/blog/area_calculation/
check_dam_height = 0.66 #metre

levels = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1,0.2, 0.3,0.4, 0.5, 0.6, 0.61, 0.7, 0.8]  #, 3.93]
cmap = cm.hot
norm = mc.BoundaryNorm(levels, cmap.N )
plt.figure(figsize=(11.69, 8.27))
CS = plt.contourf(xi, yi, zi, len(levels), alpha=.75, norm=norm, levels=levels)
C = plt.contour(xi, yi, zi, len(levels), colors='black', linewidth=.5, levels=levels)
plt.clabel(C, inline=1, fontsize=10)
plt.colorbar(CS, shrink=0.5, aspect=5)
plt.yticks(np.arange(0,30, 5))
plt.xticks(np.arange(-6,6, 2))
plt.grid()
plt.gca().invert_xaxis()
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_634/cont_2d')
plt.show()
# for i in range(len(CS.collections)):
#     print CS.levels[i]
#
# for i in range(len(C.collections)):
#     print(C.levels[i])


def contour_area(mpl_obj):
    """
    Returns a array of contour levels and
    corresponding cumulative area of contours
    :param mpl_obj: Matplotlib contour object
    :return: [(level1, area1), (level1, area1+area2)]
    """
    #Refer: Nikolai Shokhirev http://www.numericalexpert.com/blog/area_calculation/
    n_c = len(mpl_obj.collections)  # n_c = no of contours
    print 'No. of contours = %s' % n_c
    area = 0.0000
    cont_area_array = []
    for contour in range(n_c):
        # area = 0
        n_p = len(mpl_obj.collections[contour].get_paths())
        zc = mpl_obj.levels[contour]
        for path in range(n_p):
            p = mpl_obj.collections[contour].get_paths()[path]
            v = p.vertices
            l = len(v)
            s = 0.0000
            for i in range(l):
                j = (i+1) % l
                s += (v[j, 0] - v[i, 0]) * (v[j, 1] + v[i, 1])
                poly_area = 0.5*abs(s)
            area += poly_area
        cont_area_array.append((zc, area))
    return cont_area_array


# contour_area(C)
contour_a = contour_area(CS)

cont_area_df = pd.DataFrame(contour_a, columns=['Z', 'Area'])
plt.plot(cont_area_df['Z'], cont_area_df['Area'])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.ylabel(r'\textbf{Area} ($m^2$)')
plt.xlabel(r'\textbf{Stage} (m)')
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_634/cont_area_634')
# plt.show()
cont_area_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/cont_area.csv')

## Curve fitting
# fig = plt.figure(figsize=(11.69, 8.27))
y = cont_area_df['Area']
x = cont_area_df['Z']


#calculate  2nd deg polynomial
po = np.polyfit(x, y, 1)
f = np.poly1d(po)
print po
print np.poly1d(f)
#calculate new x, y
x_new = np.linspace(min(x), max(x), 50)
y_new = f(x_new)

fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(x, y, 'o', x_new, y_new)
plt.xlim([(min(x))-0.1, (max(x))+0.1])
plt.xlabel(r'\textbf{Stage} (m)')
plt.ylabel(r'\textbf{Area} ($m^2$)')
plt.text(-0.8, 500, r"$y = {0:.2f}x  {1:.2f} $".format(po[0], po[1]))
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_634/poly_2_deg_634')
plt.show()

created_profile.iloc[0] = created_profile.columns
# print created_profile
created_profile.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/created_profile_634.csv')
