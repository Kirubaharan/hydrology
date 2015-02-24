__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.interpolate import griddata
from matplotlib import cm

file_gw = '/home/kiruba/Downloads/All_arkavathy_lithologs_yield_source.csv'
df = pd.read_csv(file_gw, header=0)
X = df['Longitude_new']
Y = df['Latitude_new']
Z = df['FirstSource (m)']

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.gca(projection='3d')
xi = np.linspace(X.min(), X.max(), 100)
yi = np.linspace(Y.min(), Y.max(), 100)
zi = griddata((X,Y), Z, (xi[None, :], yi[:, None]), method='linear')
xig, yig = np.meshgrid(xi, yi)
surf = ax.scatter(xig, yig, zi, zdir='z')
fig.colorbar(surf)
plt.show()