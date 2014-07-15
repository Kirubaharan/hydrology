__author__ = 'kiruba'
import matplotlib.pyplot as plt
import pandas as pd
import pysal as ps
from matplotlib import rc
#aralumallige
aral_dbf_link = '/media/kiruba/New Volume/ACCUWA_Data/Shapefiles/microcatchment_study/milli-phaseii/check_dam/check_dam_analysis/ch_aral.dbf'
aral_dbf = ps.open(aral_dbf_link)
aral = {col:aral_dbf.by_col(col) for col in aral_dbf.header}
aral_df = pd.DataFrame(aral)
# print aral_df
# pd.options.display.mpl_style = 'default'
# print aral_df['Depth']
fig = plt.figure()
ax = fig.add_subplot(111)
aral_df['Depth'].hist(alpha=0.7,bins=[0.5, 1, 1.5, 2])
ax.grid(False)
xticks = [0, 0.5, 1, 1.5, 2.0]
ax.xaxis.set_ticks(xticks)
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'\textbf{Check Dam Height} (m)')
plt.ylabel(r'\textbf{Frequency}')
plt.title(r"Check Dams in Aralumallige watershed", fontsize=16)
# plt.show()
# plt.savefig('/media/kiruba/New Volume/ACCUWA_Data/Shapefiles/microcatchment_study/milli-phaseii/check_dam/check_dam_analysis/hist_aral')
##hadonahalli
had_dbf_link = '/media/kiruba/New Volume/ACCUWA_Data/Shapefiles/microcatchment_study/milli-phaseii/check_dam/check_dam_analysis/ch_had.dbf'
had_dbf = ps.open(had_dbf_link)
had = {col:had_dbf.by_col(col) for col in had_dbf.header}
had_df = pd.DataFrame(had)
print(had_df)
fig = plt.figure()
ax = fig.add_subplot(111)
had_df['Depth'].hist(alpha=0.7,bins=[0.5, 1, 1.5, 2])
ax.grid(False)
# xticks = [0, 0.5, 1, 1.5, 2.0]
ax.xaxis.set_ticks(xticks)
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'\textbf{Check Dam Height} (m)')
plt.ylabel(r'\textbf{Frequency}')
plt.title(r"Check Dams in Hadonahalli watershed", fontsize=16)
plt.show()
