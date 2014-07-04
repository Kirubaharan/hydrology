__author__ = 'kiruba'
##area of curve


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spread import spread
# copy the code from http://code.activestate.com/recipes/577878-generate-equally-spaced-floats/ #
import itertools
from matplotlib import rc


##read csv
csv_file = '/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/634_profile_3_sec.csv'
df = pd.read_csv(csv_file, header=0)


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2,s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

#Enter the check dam height
check_dam_height = 0.70
#to create stage with 5 cm intervals
no_of_stage_interval = check_dam_height/.05
# to create series of stage values
#dz = stage
dz = list((spread(0.00, check_dam_height, int(no_of_stage_interval), mode=3)))
# y=1
#empty list to store the results in the end
# results_1 = []
index = [range(1, 15, 1)]
columns = ['stage_m']
data = np.array(dz)
output = pd.DataFrame(data, index=index, columns=columns)
# for every value of 5 cm iteration


def calcvolume(profile, order, dy):
    """Profile = df.Y1,df.Y2,.. and order = 1,2,3"""
    results = []

    for z in dz:
        water_area = 0
        for y1, y2 in pairwise(profile):
            delev = (y2 - y1) / 10
            elev = y1
            for b in range(1, 11, 1):
                elev += delev
                if z > elev:
                    water_area += (0.1 * (z-elev))
        calc_vol = water_area * dy
        results.append(calc_vol)
    output[('Volume_%s' % order)] = results

calcvolume(df.Y1, 1, 1)
calcvolume(df.Y2, 2, 2)
calcvolume(df.Y3, 3, 2)
# print output
# add all the corresponding values
output['total_volume'] = output['Volume_1']+output['Volume_2']+output['Volume_3']
print(output)
#plot values
plt.plot(output['stage_m'], output['total_volume'], label="Stage - Volume")
plt.legend(loc='upper left')
# plt.xlabel('Stage (m)')
# plt.ylabel('Total Volume (cu.m')
# plt.title('Stage - volume relationship curve for Check Dam - 634')
##add axis labels
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'\textbf{Stage} (m)')
plt.ylabel(r'\textbf{Volume} ($m^3$)')
plt.title(r"Stage - Volume Relationship for Check Dam 634", fontsize=16)
plt.show()
plt.savefig('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/function/stage_vol_634.png')
output.to_csv('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/function/test_634.csv',sep=",")
