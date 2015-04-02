__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import checkdam.checkdam as cd
"""
Stage Volume relation estimation from survey data
"""
#input parameters
base_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/created_profile_463.csv'
check_dam_no = 463
check_dam_height = 0.66    # m
df_463 = pd.read_csv(base_file, sep=',')
# print df_616
df_463_trans = df_463.iloc[0:,2:]  # Transpose

### Renaming the column and dropping y values
y_name_list = []
for y_value in df_463_trans.columns:
    y_name_list.append(('Y_%s' %y_value))

df_463_trans.columns = y_name_list
y_value_list = df_463_trans.ix[0, 1:]

# drop the y values from data
final_data = df_463_trans.ix[1:, 0:]

#volume calculation
stage_vol_df = cd.calcvolume(y_value_list=y_value_list, elevation_data=final_data, dam_height=check_dam_height)
stage_vol_df.to_csv('/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/ch_463/stage_vol.csv')
fig = plt.figure()
plt.plot(stage_vol_df.stage_m, stage_vol_df.total_vol_cu_m)
plt.show()