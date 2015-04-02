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
# print mpl.__version__
base_file = '/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/base_profile_591.csv'
df_base = pd.read_csv(base_file, header=-1)
# print df_base.head()
# print(df_base.ix[1:, 1:])
df_base.ix[1:, 1:] = df_base.ix[1:, 1:].add(0.03)
# raise SystemExit(0)
slope_file = '/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/slope_profile_1.csv'
df_slope = pd.read_csv(slope_file, header=0)
# print df_base
df_base_trans = df_base.T    # T refers to transpose
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=36)
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
# print df_base_trans.head()
# raise SystemExit(0)
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
fig = plt.figure(figsize=(16, 8))
ax = fig.gca(projection='3d')
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
ax.set_xlabel(r'\textbf{X} (m)', fontsize=34)
ax.set_ylabel(r'\textbf{Y} (m)', fontsize=34)
ax.set_zlabel(r'\textbf{Z} (m)', fontsize=34)
# zc = CS.collections[0]
# plt.setp(zc, linewidth=4)
# plt.gca().invert_xaxis()       # invert x axis
# fig.colorbar(CS, shrink=0.5, aspect=5)  # legend
# ax = fig.add_subplot(1, 2, 2, projection='3d')
xig, yig = np.meshgrid(xi, yi)
surf = ax.plot_surface(xig, yig, zi, rstride=5, cstride=3, linewidth=0, cmap=cm.coolwarm, antialiased=False, rasterized=True)   # 3d plot
# inter_1 = []
# inter_1.append((xi, yi, zi))
# inter = pd.DataFrame(inter_1, columns=['x', 'y', 'z'])
# inter.to_csv('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/inter.csv')  # interpolation data output
fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.xlabel(r'\textbf{X} (m)')
# plt.ylabel(r'\textbf{Y} (m)')
# plt.title(r"Profile for 591", fontsize=16)
plt.gca().invert_xaxis()  # reverses x axis
# # ax = fig
# plt.savefig('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/linear_interpolation')
# html = m.fig_to_html(fig,template_type='simple')
# print html
# m.save_html(fig,'/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_591/3d_591_cd_html')
plt.show()
# raise SystemExit(0)
# ## trace contours
# Refer: Nikolai Shokhirev http://www.numericalexpert.com/blog/area_calculation/

levels = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.96] #, 2, 2.5, 3.0]
fig = plt.figure(figsize=(11.69, 8.27), facecolor='white' )
CS = plt.contourf(xi, yi, zi, len(levels), alpha=.75, cmap=cm.hot, levels=levels)
C = plt.contour(xi, yi, zi, len(levels), colors='black', linewidth=.5, levels=levels)
plt.clabel(C, inline=1, fontsize=10)
plt.colorbar(CS, shrink=0.5, aspect=5)
plt.yticks(np.arange(0,100, 5))
plt.xticks(np.arange(-30,25, 5))
plt.grid()
plt.gca().invert_xaxis()
m.save_html(fig, '/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_591/contour_html')
plt.savefig('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/cont_2d')
plt.show()
raise SystemExit(0)
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
        zc = mpl_obj.levels[contour + 1]
        print contour
        print zc
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
        print area
        cont_area_array.append((zc, area))
    return cont_area_array


# contour_area(C)
contour_a = contour_area(CS)


def poly_plot(xy, titlestr = "", margin = 0.25):
    """
        Plots polygon. For arrow see:
        http://matplotlib.org/examples/pylab_examples/arrow_simple_demo.html
        x = xy[:,0], y = xy[:,1]
    """
    xmin = np.min(xy[:,0])
    xmax = np.max(xy[:,0])
    ymin = np.min(xy[:,1])
    ymax = np.max(xy[:,1])
    hl = 0.1
    l = len(xy)
    for i in range(l):
        j = (i+1)%l  # keep index in [0,l)
        dx = xy[j,0] - xy[i,0]
        dy = xy[j,1] - xy[i,1]
        dd = np.sqrt(dx*dx + dy*dy)
        dx = dx*(1 - hl/dd)
        dy = dy*(1 - hl/dd)
        plt.arrow(xy[i,0], xy[i,1], dx, dy, head_width=0.05, head_length=0.1, fc='b', ec='b')
        plt.xlim(xmin-margin, xmax+margin)
        plt.ylim(ymin-margin, ymax+margin)
    plt.title(titlestr)


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

# # zero contour has two paths 0, 1
# p_0_0 = CS.collections[0].get_paths()[0]    # CS.collections[index of contour].get_paths()[index of path]
# p_0_1 = CS.collections[0].get_paths()[1]
# v_0_0 = p_0_0.vertices
# v_0_1 = p_0_1.vertices
# area_0_0 = abs(poly_area(v_0_0))
# area_0_1 = abs(poly_area(v_0_1))
# area_0 = area_0_0 + area_0_1
# z_0 = CS.levels[0]
# # print z_0, area_0
# plt.fill(v_0_0[:,0], v_0_0[:,1], facecolor='g')
# plt.show()
# # 0.4 contour has three paths 0,1,2
# print len(CS.collections[21].get_paths())
print(levels)
# print(CS.levels[22])
print len(CS.levels)

# p_1_0 = CS.collections[21].get_paths()[0]
# p_1_1 = CS.collections[21].get_paths()[1]
# # p_1_2 = CS.collections[21].get_paths()[2]
# v_1_0 = p_1_0.vertices
# v_1_1 = p_1_1.vertices
# v_1_2 = p_1_2.vertices
# area_1_0 = poly_area(v_1_0)
# print(area_1_0)
# area_1_1 = poly_area(v_1_1)
# area_1_2 = poly_area(v_1_2)
# print area_1_1, area_1_2
# area_1 = area_1_0 + area_1_1 + area_1_2
# z_1 = CS.levels[1]
# print z_1, area_1
# fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
# plt.fill(v_1_0[:,0], v_1_0[:,1], facecolor='r')
# plt.yticks(np.arange(0,100, 5))
# plt.xticks(np.arange(-30,25, 5))
# plt.grid()
# plt.gca().invert_xaxis()
# # plt.show()
# fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
# plt.fill(v_1_1[:,0], v_1_1[:,1], facecolor='b')
# plt.yticks(np.arange(0,100, 5))
# plt.xticks(np.arange(-30,25, 5))
# plt.grid()
# plt.gca().invert_xaxis()
# plt.show()
# fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
# plt.fill(v_1_2[:,0], v_1_2[:,1], facecolor='g')
# plt.yticks(np.arange(0,100, 5))
# plt.xticks(np.arange(-30,25, 5))
# plt.grid()
# plt.gca().invert_xaxis()
plt.show()
area = 0.0


def areaofpolygon(polygon, i):
    global area
    if i == 0:
        area = 0
    try:
        x1, y1 = polygon[i]
        x2, y2 = polygon[i+1]
        area += (x1*y2) - (x2*y1)
    except IndexError, e:
        x1, y1 = polygon[0]
        x2, y2 = polygon[-1]
        area += (x2*y1) - (x1*y2)
        return abs(area/2.0)
    return areaofpolygon(polygon, i+1)

# new_area_0 = areaofpolygon(v_1_0, 0)
# new_area_1 = areaofpolygon(v_1_1, 0)
# new_area_2 = areaofpolygon(v_1_2, 0)
# print new_area_0, new_area_1, new_area_2
# # 0.8 contour has three paths 0,1,2
# print len(CS.collections[2].get_paths())
# p_2_0 = CS.collections[2].get_paths()[0]
# p_2_1 = CS.collections[2].get_paths()[1]
# p_2_2 = CS.collections[2].get_paths()[2]
# v_2_0 = p_2_0.vertices
# v_2_1 = p_2_1.vertices
# v_2_2 = p_2_2.vertices
# area_2_0 = poly_area(v_2_0)
# print(area_2_0)
# area_2_1 = poly_area(v_2_1)
# area_2_2 = poly_area(v_2_2)
# print(area_2_1, area_2_2)
# area_2 = area_2_0 + area_2_1 + area_2_2
# z_2 = CS.levels[2]
# print z_2, area_2
# plt.fill(v_2_0[:,0], v_2_0[:,1], facecolor='b')
# plt.show()
# poly_plot(v_2_0, titlestr="1st path")
# plt.show()
# poly_plot(v_2_1, titlestr="2nd path")
# plt.show()
# poly_plot(v_2_2, titlestr="3rd path")
# plt.show()



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
print cont_area_df

plt.plot(cont_area_df['Z'], cont_area_df['Area'])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.ylabel(r'\textbf{Area} ($m^2$)')
plt.xlabel(r'\textbf{Stage} (m)')
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/cont_area_591')
# plt.show()
cont_area_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/cont_area.csv')

## Curve fitting
# fig = plt.figure(figsize=(11.69, 8.27))
y = cont_area_df['Area']
x = cont_area_df['Z']

# calculate linear fit
# po = np.polyfit(x, y, 1)
# f = np.poly1d(po)
# print np.poly1d(po)
# x_new = np.linspace(min(x), max(x), 50)
# y_new = f(x_new)
# plt.plot(x,y, 'o', x_new, y_new)
# plt.xlim([(min(x))-1, (max(x))+1])
# plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/linear_fit_591')
# plt.show()

#calculate  2nd deg polynomial
po = np.polyfit(x, y, 2)
f = np.poly1d(po)
# print po
# print np.poly1d(f)
#calculate new x, y
x_new = np.linspace(min(x), max(x), 50)
y_new = f(x_new)

fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(x, y, 'o', x_new, y_new)
plt.xlim([(min(x))-1, (max(x))+1])
plt.xlabel(r'\textbf{Stage} (m)')
plt.ylabel(r'\textbf{Area} ($m^2$)')
plt.text(-0.8, 500, r"$y = {0:.2f}x^2 {1:.2f}x + {2:.2f}$".format(po[0], po[1], po[2]))
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_evap/poly_2_deg_591')
# plt.show()


def set_column_sequence(dataframe, seq):
    '''Takes a dataframe and a subsequence of its columns, returns dataframe with seq as first columns'''
    cols = seq[:] # copy so we don't mutate seq
    for x in dataframe.columns:
        if x not in cols:
            cols.append(x)
    return dataframe[cols]

raise SystemExit(0)
df_base_trans.columns = df_base_trans.iloc[0, 0:]
# print df_base_trans
sorted_df = df_base_trans.iloc[1:, 1:]
sorted_df = sorted_df[sorted(sorted_df.columns)]
# print sorted_df
# print df_base_trans.iloc[0:, 0]
sorted_df = sorted_df.join(df_base_trans.iloc[0:, 0], how='left')
df_base_trans = set_column_sequence(sorted_df, [1500])
df_base_trans.iloc[0] = df_base_trans.columns
# print df_base_trans
df_base_trans.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/created_profile_591.csv')
