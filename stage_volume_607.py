__author__ = 'kiruba'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spread import spread
# copy the code from http://code.activestate.com/recipes/577878-generate-equally-spaced-floats/ #
import itertools
from matplotlib import rc

#function to create iterator series


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2,s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

#function to create stage volume output

def calcvolume(profile, order, dy):
    """
    Profile = df.Y1,df.Y2,.. and order = 1,2,3
    :param profile: series of Y values
    :param order: distance from origin
    :param dy: thickness of profile in m
    :return: volume for profile
    """

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
#inout parameters
check_dam_no = 607
check_dam_height = 2
no_of_stage_interval = check_dam_height/.05

#create a list of stage values with 5 cm interval
dz = list((spread(0.00, check_dam_height, int(no_of_stage_interval), mode=3)))

#surveyed data
input_file = '/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/created_profile_607.csv'
df = pd.read_csv(input_file, header=0)
row = 17   # row of Y values

index = [range(1, 42, 1)]  # no of stage intervals
columns = ['stage_m']
data = np.array(dz)
output = pd.DataFrame(data, index=index, columns=columns)  # dataframe with stage values

for l1, l2 in pairwise(df.ix[row]):
    if l2 > 0:
        calcvolume(profile=df["Y_%s" % int(l1)], order=l1, dy=int(l2-l1))

output_series = output.filter(regex="Volume_")  # filter the columns that have Volume_
output["total_vol"] = output_series.sum(axis=1)  # get total volume
print output.head(5)
output.to_csv('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/stage_vol_607.csv', sep=",")  #output file
### Plotting
plt.plot(output['stage_m'], output['total_vol'], label="Stage - Volume")
plt.legend(loc='upper left')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'\textbf{Stage} (m)')
plt.ylabel(r'\textbf{Volume} ($m^3$)')
plt.title(r"Stage - Volume Relationship for Check Dam - %s" % check_dam_no, fontsize=16)
plt.savefig('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/function/stage_vol_607')
plt.show()