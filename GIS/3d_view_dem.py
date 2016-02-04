__author__ = 'kiruba'
import matplotlib
matplotlib.use('QT4Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import gdal
from mpl_toolkits.mplot3d import Axes3D, axes3d
from mayavi import mlab
from scipy.interpolate import griddata
from matplotlib import cm
import matplotlib.rcsetup as rcsetup
print(rcsetup.all_backends)



data = np.loadtxt('/media/kiruba/New Volume/milli_watershed/Hadonahalli_dem/had_dem_utm.txt', delimiter=',')
print data
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

xi = np.linspace(x.min(), x.max(), 500)
yi = np.linspace(y.min(), y.max(), 500)
zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
xig, yig = np.meshgrid(xi, yi)
print "ok"
# mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1,1,1))
# mlab.mesh(xig, yig, zi, representation='wireframe')
# mlab.show()

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.gca(projection='3d')
# ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xig, yig, zi, rstride=30, cstride=40, linewidth=0, cmap=cm.coolwarm)
# boundary = plt.plot(boundary_df['longitude'], boundary_df['latitude'], 'r-o')
# points = plt.plot(dt_bathymetry_df['longitude'], dt_bathymetry_df['latitude'], 'ko')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
# mlab.figure(size=(640, 800), bgcolor=(0.16, 0.28, 0.46))
# mlab.surf(values, warp_scale=0.2)
# mlab.show()