__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import checkdam.checkdam as cd

"""
Stage Volume relation estimation from survey data
"""
#input parameters
base_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/created_profile_634.csv'
check_dam_no = 634
check_dam_height = 0.64    # m
df_634 = pd.read_csv(base_file, sep=',')
# print df_616
df_634_trans = df_634.iloc[0:,2:]  # Transpose

# print(df_59_trans)
# print len(df_591_trans.ix[1:, 0])
### Renaming the column and dropping y values
y_name_list = []
for y_value in df_634_trans.columns:
    y_name_list.append(('Y_%s' %y_value))

df_634_trans.columns = y_name_list
# print df_591_trans
y_value_list = df_634_trans.ix[0, 1:]
# print y_value_list

# drop the y values from data
final_data = df_634_trans.ix[1:, 0:]
# print final_data

stage_vol_df = cd.calcvolume(y_value_list=y_value_list,elevation_data=final_data, dam_height=check_dam_height)
print stage_vol_df
stage_vol_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_634/stage_vol.csv')
fig = plt.figure()
plt.plot(stage_vol_df.stage_m, stage_vol_df.total_vol_cu_m)
plt.show()
#
#
# def calcvolume2(y_value_list, elevation_data, dam_height):
#     """
#
#     :param y_value_list:
#     :param profile:
#     :param dam_height:
#     :return:
#     """
#     no_of_stage_interval = dam_height/0.05
#     dz = list((spread(0.00, check_dam_height, int(no_of_stage_interval), mode=3)))
#     index = [range(len(dz))]  # no of stage intervals
#     columns = ['stage_m']
#     data = np.array(dz)
#     output = pd.DataFrame(data, index=index, columns=columns)
#     for l1, l2 in pairwise(y_value_list):
#         results = []
#         profile=elevation_data["Y_%s" % float(l1)]
#         order = l1
#         dy = int(l2-l1)
#         for stage in dz:
#             water_area = 0
#             for z1, z2 in pairwise(profile):
#                 delev = (z2 - z1) / 10
#                 elev = z1
#                 for b in range(1, 11, 1):
#                     elev += delev
#                     if stage > elev:
#                         water_area += (0.1 * (stage-elev))
#
#                 calc_vol_2 = water_area*dy
#             results.append(calc_vol_2)
#
#         output[('Volume_%s' % order)] = results
#
#     output_series = output.filter(regex="Volume_")
#     output["total_vol_cu_m"] = output_series.sum(axis=1)
#     return output[['stage_m', "total_vol_cu_m"]]
#
#
# stage_vol_df_1 = calcvolume2(y_value_list=y_value_list, elevation_data=final_data, dam_height=check_dam_height)
#
# print stage_vol_df_1
# fig = plt.figure()
# plt.plot(stage_vol_df_1.stage_m, stage_vol_df_1.total_vol_cu_m)
# plt.show()