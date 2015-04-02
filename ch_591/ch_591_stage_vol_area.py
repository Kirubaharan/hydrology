__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import checkdam.checkdam as cd
"""
Stage Volume relation estimation from survey data
"""
#input parameters
base_file_591 = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/created_profile_591.csv'
check_dam_no = 591
check_dam_height = 1.96    # m
df_591 = pd.read_csv(base_file_591, sep=',')
# print df_591
df_591_trans = df_591.iloc[0:, 2:]  # Transpose
### Renaming the column and dropping y values
# print df_591_trans
y_name_list = []
for y_value in df_591_trans.columns:
    y_name_list.append(('Y_%s' %y_value))

df_591_trans.columns = y_name_list
# print df_591_trans
y_value_list = df_591_trans.ix[0, 1:]
# print y_value_list

# drop the y values from data
final_data = df_591_trans.ix[1:, 0:]
# print final_data

stage_vol_df = cd.calcvolume(y_value_list=y_value_list, elevation_data=final_data, dam_height=1.96)
stage_vol_df['stage_m'] = cd.myround(stage_vol_df['stage_m'], decimals=2)
print stage_vol_df
# raise SystemExit(0)
stage_vol_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/stage_vol_new.csv')
# fig=plt.figure()
# plt.plot(stage_vol_df.stage_m, stage_vol_df.total_vol_cu_m)
# plt.show()