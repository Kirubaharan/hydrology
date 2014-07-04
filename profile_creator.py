__author__ = 'kiruba'

# import statements
import matplotlib.pyplot as plt
# import  numpy as np
import pandas as pd
from matplotlib import rc
import itertools
# from mpl_toolkits.mplot3d import axes3d, Axes3D
# from matplotlib import cm

slope_profile_csv = '/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/slope_profile_607.csv'
df_profile = pd.read_csv(slope_profile_csv, header=0)
base_profile_csv = '/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/base_profile_607.csv'
df_base = pd.read_csv(base_profile_csv, header=0)

##plot
## function to create pairs of iterable elevations


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2,s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

template = df_base.Y_11
z1 = 0.27

data = []
for x in df_base.X:
    for y, z2 in df_profile.itertuples(index=False):
        diff = z2 - z1
        new_profile = template + diff
        df_base[('Y_%s' % y)] = new_profile
        for i in new_profile:
            data.append((x, y, i))
    df_base.to_csv('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/created_profile_607.csv')
data_df = pd.DataFrame(data, columns=['x', 'y', 'z'])
data_df.to_csv('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/test_data.csv')

# print data_df
X = data_df.x
Y = data_df.y
Z = data_df.z

fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# ax.plot_wireframe(X,Y,Z, rstride=100, cstride=50)
ax = fig.gca(projection='3d')
# ax.plot_trisurf(X,Y,Z,cmap =cm.jet,linewidth =0.2)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
# cset = ax.contour(X,Y,Z, zdir='z', offset=-100, cmap=cm.coolwarm)
# cset = ax.contour(X,Y,Z, zdir='x', offset=-40, cmap=cm.coolwarm)
# cset = ax.contour(X,Y,Z, zdir='y', offset=40, cmap=cm.coolwarm)

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'\textbf{X} (m)')
plt.ylabel(r'\textbf{Y} (m)')
plt.title(r"Profile for 607", fontsize=16)
plt.show()
plt.savefig('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/607_created_profile')
