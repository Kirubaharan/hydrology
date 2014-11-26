__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from spread import spread
"""
Stage Volume relation estimation from survey data
"""
# neccessary functions


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2,s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)
#function to create stage volume output


def calcvolume(profile, order, dy):
    """
    Profile = df.Y1,df.Y2,.. and order = 1,2,3
    :param profile: series of Z values
    :param order: distance from origin
    :param dy: thickness of profile in m
    :param dam_height: Height of check dam in m
    :return: output: pandas dataframe volume for profile
    """

    # print 'profile length = %s' % len(profile)
    results = []

    for stage in dz:
        water_area = 0
        for z1, z2 in pairwise(profile):
            delev = (z2 - z1) / 10
            elev = z1
            for b in range(1, 11, 1):
                elev += delev
                if stage > elev:
                    # print 'elev = %s' % elev
                    water_area += (0.1 * (stage-elev))
                    # print 'order = %s and dy = %s' %(order, dy)
                    # print 'area = %s' % water_area

            calc_vol = water_area * dy
        # print 'calc vol = %s' % calc_vol
        results.append(calc_vol)
        # print 'results = %s' % results
        # print 'results length = %s' % len(results)

    output[('Volume_%s' % order)] = results
#input parameters
base_file_591 = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/created_profile_591.csv'
check_dam_no = 591
check_dam_height = 1.9    # m
df_591 = pd.read_csv(base_file_591, sep=',')
print df_591
df_591_trans = df_591.iloc[0:, 2:]  # Transpose
no_of_stage_interval = check_dam_height/.05
dz = list((spread(0.00, check_dam_height, int(no_of_stage_interval), mode=3)))
index = [range(len(dz))]  # no of stage intervals
columns = ['stage_m']
data = np.array(dz)
output = pd.DataFrame(data, index=index, columns=columns)
print(df_591_trans)
# print len(df_591_trans.ix[1:, 0])
### Renaming the column and dropping y values
# print df_591_trans
y_name_list = []
for y_value in df_591_trans.columns:
    y_name_list.append(('Y_%s' %y_value))

df_591_trans.columns = y_name_list
# print df_591_trans
y_value_list = df_591_trans.ix[0, 1:]
print y_value_list

# drop the y values from data
final_data = df_591_trans.ix[1:, 0:]
print final_data

#volume calculation
for l1, l2 in pairwise(y_value_list):
    calcvolume(profile=final_data["Y_%s" % float(l1)], order=l1, dy=int(l2-l1))

output_series = output.filter(regex="Volume_")  # filter the columns that have Volume_
output["total_vol_cu_m"] = output_series.sum(axis=1)  # get total volume
# print output

# select only stage and total volume
stage_vol_df = output[['stage_m', "total_vol_cu_m"]]

stage_vol_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/stage_vol_new.csv')
fig=plt.figure()
plt.plot(stage_vol_df.stage_m, stage_vol_df.total_vol_cu_m)
plt.show()