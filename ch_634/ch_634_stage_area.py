__author__ = 'kiruba'
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.path import *
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib as mpl
import matplotlib.colors as mc
import checkdam.checkdam as cd

# latex parameters
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)

base_file = '/media/kiruba/New Volume/milli_watershed/stream_profile/634/base_profile_634.csv'
df_base = pd.read_csv(base_file, header=-1, skiprows=1)
# correction
df_base.ix[1,5] = -0.02
df_base.ix[2,5] = -0.03
# print df_base.head()
df_base.ix[1:,1:] = df_base.ix[1:, 1:].add(0.03)
# print df_base.head()
df_base_trans = df_base.T
df_base_trans.columns = df_base_trans.ix[0, 0:]
# print df_base_trans
df_base_trans = df_base_trans.ix[1:, 1500:]


# created_profile = created_profile[sorted(created_profile.columns)]
created_profile = df_base_trans
# print created_profile.head()
sorted_df = created_profile.iloc[0:, 1:]
sorted_df = sorted_df[sorted(sorted_df.columns)]
sorted_df = sorted_df.join(created_profile.iloc[0:, 0], how='right')
created_profile = cd.set_column_sequence(sorted_df, [1500])
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
check_dam_height = 0.64 #metre

levels = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1,0.2, 0.3,0.4, 0.5, 0.6, 0.64, 0.7, 0.8]  #, 3.93]
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

contour_a = cd.contour_area(CS)

cont_area_df = pd.DataFrame(contour_a, columns=['Z', 'Area'])
plt.plot(cont_area_df['Z'], cont_area_df['Area'])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.ylabel(r'\textbf{Area} ($m^2$)')
plt.xlabel(r'\textbf{Stage} (m)')
plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/python_plots/check_dam_634/cont_area_634')
plt.show()
cont_area_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/cont_area.csv')

created_profile.iloc[0] = created_profile.columns
# print created_profile
created_profile.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/created_profile_634.csv')
