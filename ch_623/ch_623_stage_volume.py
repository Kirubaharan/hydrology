__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import checkdam.checkdam as cd
"""
Stage Volume relation estimation from survey data
"""
#input parameters
base_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_623/created_profile_623.csv'
check_dam_no = 623
check_dam_height = 1.41    # m
df_623 = pd.read_csv(base_file, sep=',')
# print df_616
df_623_trans = df_623.iloc[0:,2:]  # Transpose

### Renaming the column and dropping y values
y_name_list = []
for y_value in df_623_trans.columns:
    y_name_list.append(('Y_%s' %y_value))

df_623_trans.columns = y_name_list
# print df_591_trans
y_value_list = df_623_trans.ix[0, 1:]
# print y_value_list

# drop the y values from data
final_data = df_623_trans.ix[1:, 0:]
# print final_data

#volume calculation


# select only stage and total volume
stage_vol_df = cd.calcvolume(y_value_list=y_value_list, elevation_data=final_data, dam_height=check_dam_height)
print stage_vol_df
stage_vol_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_623/stage_vol.csv')
fig = plt.figure()
plt.plot(stage_vol_df.stage_m, stage_vol_df.total_vol_cu_m)
plt.show()