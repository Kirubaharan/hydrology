__author__ = 'kiruba'
from pairwise import pairwise
from checkdam import calcvolume
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import rc
from mayavi import mlab
from numpy import linspace, meshgrid
from scipy.interpolate import griddata
import numpy as np
import matplotlib.mlab as ml

base_file = '/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/base_profile_591.csv'
df_base = pd.read_csv(base_file, header=-1)
slope_file = '/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/591/slope_profile_1.csv'
df_slope = pd.read_csv(slope_file, header=0)
# print df_base
df_base_trans = df_base.T
# print(df_base_trans)
# print df_slope
# # df_base 17(0-16) rows and 47(0-46) columns
# df_base_trans  has 47 (0-46) rows and 17(0-16) columns

# print df_base_trans.ix[1:, 1]   # df.ix[row,column]
template_0 = df_base_trans.ix[1:, 1]
# print template_0
# print df_base_trans.ix[0:, 0:1]
z1 = df_base_trans.ix[29, 1]
z2 = .21   # elevation value at 9 m
diff = z2 - z1
profile_9 = template_0 + diff
df_base_trans[17] = profile_9
df_base_trans.ix[0,17] = 9

template_10 = df_base_trans.ix[1:, 2]
z10 = df_base_trans.ix[29, 2]
z11  = .11
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

x1 = df_base_trans.ix[1:, 0]
y1 = df_base_trans.ix[0, 1:]
z1 = df_base_trans.ix[1:, 1:]

z_array = df_base_trans.ix[1:, 1:].values
columns = list(df_base_trans.ix[0, 1:].values)
index = df_base_trans.ix[1:, 0].values
df = pd.DataFrame(z_array, columns=columns).set_index(index)

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

# print data_1_df

X = data_1_df.x
Y = data_1_df.y
Z = data_1_df.z
#
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection = '3d')
ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
# ax.plot_surface(X,Y,Z, rstride=4, cstride=4, linewidth=2)
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'\textbf{X} (m)')
plt.ylabel(r'\textbf{Y} (m)')
plt.title(r"Profile for 607", fontsize=16)
# plt.show()

xmin, xmax, ymin, ymax =[min(X), max(X), min(Y), max(Y)]
nx = (int(xmax-xmin+1))
ny = (int(ymax-ymin+1))
print nx, ny

xi = linspace(xmin,xmax,nx)
yi = linspace(ymin,ymax,ny)
xi,yi = meshgrid(xi,yi)
ml.griddata(X,Y,Z,xi,yi,interp='nn')

# mlab.points3d(X1, Y1, Z1, Z1)

# points = np.random.rand(1000,2)
# print points









#
# # pts = mlab.points3d(X, Y, Z)
#
# pts = mlab.points3d(X, Y, Z, Z)
# #
# mesh = mlab.pipeline.delaunay2d(pts)
# #
# pts.remove()
# #
# surf = mlab.pipeline.surface(mesh)
# mlab.contour3d(pts)
mlab.show()


