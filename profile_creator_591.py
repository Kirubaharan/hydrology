__author__ = 'kiruba'
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import rc
from scipy.interpolate import griddata
import numpy as np
from matplotlib import cm
from matplotlib.path import *

base_file = '/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/base_profile_591.csv'
df_base = pd.read_csv(base_file, header=-1)
slope_file = '/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/slope_profile_1.csv'
df_slope = pd.read_csv(slope_file, header=0)
# print df_base
df_base_trans = df_base.T
# print(df_base_trans)
# print df_slope
# check dam height = 1.9 m
#width - 8.8 m
# # df_base 17(0-16) rows and 47(0-46) columns
# df_base_trans  has 47 (0-46) rows and 17(0-16) columns
############################################ In between profiles are filled
# print df_base_trans.ix[1:, 1]   # df.ix[row,column]
template_0 = df_base_trans.ix[1:, 1]  # template for profile
# print template_0
# print df_base_trans.ix[0:, 0:1]
z1 = df_base_trans.ix[29, 1]
z2 = .21   # elevation value at 9 m
diff = z2 - z1
profile_9 = template_0 + diff
df_base_trans[17] = profile_9
df_base_trans.ix[0, 17] = 9

template_10 = df_base_trans.ix[1:, 2]
z10 = df_base_trans.ix[29, 2]
z11 = .11
diff = z11 - z10
profile_11 = template_10 + diff
df_base_trans[18] = profile_11
df_base_trans.ix[0, 18] = 11

template_20 = df_base_trans.ix[1:, 3]
z20 = df_base_trans.ix[29, 3]
z28 = 0.49
diff = z28 - z20
profile_28 = template_20 + diff
df_base_trans[19] = profile_28
df_base_trans.ix[0, 19] = 28

template_30 = df_base_trans.ix[1:, 4]
z30 = df_base_trans.ix[29, 4]
z38 = .66
diff = z38 - z30
profile_38 = template_30 + diff
df_base_trans[20] = profile_38
df_base_trans.ix[0, 20] = 38
# print df_base_trans
################################################
x1 = df_base_trans.ix[1:, 0]   # separate out x, y, z values
y1 = df_base_trans.ix[0, 1:]
z1 = df_base_trans.ix[1:, 1:]

z_array = df_base_trans.ix[1:, 1:].values
columns = list(df_base_trans.ix[0, 1:].values)
index = df_base_trans.ix[1:, 0].values
df = pd.DataFrame(z_array, columns=columns).set_index(index)
#### create x, y, z array for plotting and contour
# print df
data_1 = []
for y, row in df.iteritems():
    # print 'i = %s' % y
    for x, z in row.iteritems():
        data_1.append((x, y, z))
        # print 'x = %s and z = %s' % (x,z)

data_1_df = pd.DataFrame(data_1, columns=['x', 'y', 'z'])
# df_base_trans.to_csv('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/base_trans.csv')
data_1_df.to_csv('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/cloud.csv')

# print data_1_df.shape

X = data_1_df.x
Y = data_1_df.y
Z = data_1_df.z
#
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection = '3d')
# ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
# ax.plot_surface(X,Y,Z, rstride=4, cstride=4, linewidth=2)
# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# rc('text', usetex=True)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.xlabel(r'\textbf{X} (m)')
# plt.ylabel(r'\textbf{Y} (m)')
# plt.title(r"Profile for 591", fontsize=16)
# plt.show()

## contour and 3d surface plotting
# fig = plt.figure(figsize=plt.figaspect(0.5))
# ax = fig.add_subplot(1, 2, 1, projection='3d')
xi = np.linspace(X.min(), X.max(), 100)
yi = np.linspace(Y.min(), Y.max(), 100)
# print len(xi)
# print len(yi)
# print len(Z)
zi = griddata((X, Y), Z, (xi[None, :], yi[:, None]), method='linear')    # create a uniform spaced grid
# print zi.min()
# print zi.max()
# CS_1 = plt.contourf(xi, yi, zi, 36, alpha =.75, cmap= 'jet')
# CS = plt.contour(xi, yi, zi, 36, linewidths=0.5, color='black')       # contour with .1 m interval
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.xlabel(r'\textbf{X} (m)')
# plt.ylabel(r'\textbf{Y} (m)')
# zc = CS.collections[0]
# plt.setp(zc, linewidth=4)
# plt.gca().invert_xaxis()       # invert x axis
# fig.colorbar(CS, shrink=0.5, aspect=5)  # legend
# ax = fig.add_subplot(1, 2, 2, projection='3d')
# xig, yig = np.meshgrid(xi, yi)
# surf = ax.plot_surface(xig, yig, zi, rstride=5, cstride=3, linewidth=0, cmap=cm.coolwarm, antialiased=False)   # 3d plot
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
# plt.gca().invert_xaxis()  # reverses x axis
# # ax = fig
# plt.savefig('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/linear_interpolation')
# plt.show()

# ## trace contours
# Refer: Nikolai Shokhirev http://www.numericalexpert.com/blog/area_calculation/

levels = [0, 0.4, 0.8, 1.2, 1.4, 1.6, 1.9, 2.4]  #, 3.93]
plt.figure(figsize=(11.69, 8.27))
CS = plt.contourf(xi, yi, zi, len(levels),alpha=.75, cmap=cm.hot, levels=levels)
C = plt.contour(xi, yi, zi, len(levels), colors='black', linewidth=.5, levels=levels)
plt.clabel(C, inline=1, fontsize=10)
plt.colorbar(CS, shrink=0.5, aspect=5)
plt.grid()
plt.savefig('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/cont_2d')
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
    area = 0
    cont_area_array = []
    for contour in range(n_c):
        # area = 0
        n_p = len(mpl_obj.collections[contour].get_paths())
        zc = mpl_obj.levels[contour]
        for path in range(n_p):
            p = mpl_obj.collections[contour].get_paths()[path]
            v = p.vertices
            l = len(v)
            s = 0
            for i in range(l):
                j = (i+1) % l
                s += (v[j, 0] - v[i, 0]) * (v[j, 1] + v[i, 1])
                poly_area = abs(-0.5*s)
            area += poly_area
        cont_area_array.append((zc, area))
    return cont_area_array


# contour_area(C)
contour_a = contour_area(CS)

# zero contour has two paths 0, 1
# p_0_0 = CS.collections[0].get_paths()[0]    # CS.collections[index of contour].get_paths()[index of path]
# p_0_1 = CS.collections[0].get_paths()[1]
# v_0_0 = p_0_0.vertices
# v_0_1 = p_0_1.vertices
# area_0_0 = abs(poly_area(v_0_0))
# area_0_1 = abs(poly_area(v_0_1))
# area_0 = area_0_0 + area_0_1
# z_0 = CS.levels[0]
# print z_0, area_0

# 0.4 contour has three paths 0,1,2
# p_1_0 = CS.collections[1].get_paths()[0]
# p_1_1 = CS.collections[1].get_paths()[1]
# p_1_2 = CS.collections[1].get_paths()[2]
# v_1_0 = p_1_0.vertices
# v_1_1 = p_1_1.vertices
# v_1_2 = p_1_2.vertices
# area_1_0 = poly_area(v_1_0)
# area_1_1 = poly_area(v_1_1)
# area_1_2 = poly_area(v_1_2)
# area_1 = area_1_0 + area_1_1 + area_1_2
# z_1 = CS.levels[1]
# print z_1, area_1

# 0.8 contour has three paths 0,1,2
# p_2_0 = CS.collections[2].get_paths()[0]
# p_2_1 = CS.collections[2].get_paths()[1]
# p_2_2 = CS.collections[2].get_paths()[2]
# v_2_0 = p_2_0.vertices
# v_2_1 = p_2_1.vertices
# v_2_2 = p_2_2.vertices
# area_2_0 = poly_area(v_2_0)
# area_2_1 = poly_area(v_2_1)
# area_2_2 = poly_area(v_2_2)
# area_2 = area_2_0 + area_2_1 + area_2_2
# z_2 = CS.levels[2]
# print z_2, area_2

# 0.8 contour has two paths 0,1
# p_3_0 = CS.collections[3].get_paths()[0]
# p_3_1 = CS.collections[3].get_paths()[1]
# v_3_0 = p_3_0.vertices
# v_3_1 = p_3_1.vertices
# area_3_0 = abs(poly_area(v_3_0))
# area_3_1 = abs(poly_area(v_3_1))
# area_3 = area_3_0 + area_3_1
# z_3 = CS.levels[3]
# print z_3, area_3


cont_area_df = pd.DataFrame(contour_a, columns=['Z', 'Area'])
plt.plot( cont_area_df['Area'], cont_area_df['Z'])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'\textbf{Area} ($m^2$)')
plt.ylabel(r'\textbf{Stage} (m)')
plt.savefig('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/cont_area')
plt.show()
cont_area_df.to_csv('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/cont_area.csv')

