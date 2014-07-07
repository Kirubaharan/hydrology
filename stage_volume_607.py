__author__ = 'kiruba'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spread import spread
# copy the code from http://code.activestate.com/recipes/577878-generate-equally-spaced-floats/ #
import itertools
from matplotlib import rc


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2,s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


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
# print(df.ix[row])
# index = df['']


check_dam_height = 2
no_of_stage_interval = check_dam_height/.05

dz = list((spread(0.00, check_dam_height, int(no_of_stage_interval), mode=3)))
# print dz

input_file = '/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/created_profile_607.csv'
df = pd.read_csv(input_file, header=0)
row = 17

index = [range(1, 42, 1)]
columns = ['stage_m']
data = np.array(dz)
output = pd.DataFrame(data, index=index, columns=columns)

for l1, l2 in pairwise(df.ix[row]):
    if l2 > 0:
        print("Processing for %s" % l1)
        for idx, row in df.iteritems():
            if idx != "Unnamed":
                profile_name = "df." +idx
                print profile_name
                print row
                calcvolume(profile=profile_name,order=l1,dy=int(l2-l1))

print output