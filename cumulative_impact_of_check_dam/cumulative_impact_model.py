__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from contextlib import contextmanager
import simpy

"""
Parameters
"""
INFLOW_463 = [0, 0, 10, 5, 0, 10]
INFLOW_640 = [0, 0, 8, 3, 0, 8]
EVAP_463 = [0, 0, 1, 1, 1, 1]
EVAP_640 = [0, 0, 1, 1, 1, 1]
TOTAL_VOL_463 = 8.5
TOTAL_VOL_640 = 5
INITIAL_VOL = 0
TOTAL_DAYS = len(INFLOW_463)
print TOTAL_DAYS


@contextmanager
def columns_in(df):
    """
    Source: https://gist.github.com/aflaxman/4121076#file-sdm_diaper_delivery-ipynb
    :param df: Pandas dataframe
    :return: column names in df as a tuple
    """
    col_names = df.columns
    col_list = [df[col] for col in col_names]
    try:
        yield tuple(col_list)
    finally:
        for i, col, in enumerate(col_names):
            df[col] = col_list[i]


column_names = ['current_volume_463', 'current_volume_640', 'inflow_463', 'inflow_640', 'evap_463', 'evap_640',
                'overflow_463', 'overflow_640']
state = pd.DataFrame(index=range(-7, TOTAL_DAYS), columns=column_names)
state.ix[-7:0, ['current_volume_463', 'current_volume_640']] = INITIAL_VOL
state.ix[-7:0, ['inflow_463', 'inflow_640', 'evap_463', 'evap_640']] = 0
state.ix[0:TOTAL_DAYS - 1, ['inflow_463']] = INFLOW_463
state.ix[0:TOTAL_DAYS - 1, ['inflow_640']] = INFLOW_640
state.ix[0:TOTAL_DAYS - 1, ['evap_463']] = EVAP_463
state.ix[0:TOTAL_DAYS - 1, ['evap_640']] = EVAP_640

with columns_in(state) as (current_volume_463, current_volume_640, inflow_463, inflow_640, evap_463, evap_640, overflow_463, overflow_640):
    for t in range(TOTAL_DAYS - 1):
        current_volume_463[t + 1] = current_volume_463[t] + inflow_463[t] - evap_463[t]
        current_volume_640[t + 1] = current_volume_640[t] + inflow_640[t] - evap_640[t]
        if current_volume_463[t + 1] < 0:
            current_volume_463[t + 1] = 0
        if current_volume_640[t + 1] < 0:
            current_volume_640[t + 1] = 0
        if current_volume_463[t + 1] > TOTAL_VOL_463:
            overflow_463[t + 1] = current_volume_463[t+1] - TOTAL_VOL_463
            current_volume_463[t+1] = TOTAL_VOL_463
        if current_volume_640[t + 1] > TOTAL_VOL_640:
            overflow_640[t + 1] = current_volume_640[t+1] - TOTAL_VOL_640
            current_volume_640[t+1] = TOTAL_VOL_640

t = state.filter(['current_volume_463', 'current_volume_640', 'inflow_463', 'inflow_640', 'overflow_463', 'overflow_640'])
t.plot(linewidth=2, alpha=0.9)
plt.show()


class Catchment_Parameters(object):
    def __init__(self,name, rainfall, evaporation, infiltration_rate):
        self.name = name

    def rainfall_array(self, rainfall):


class Checkdam(Catchment_Parameters):
    def __init__(self, max_volume, stage_area, stage_volume, initial_volume, catchment_area):
        self.max_volume = max_volume
        self.stage_area = stage_area
        self.stage_volume = stage_volume
        self.initial_volume = initial_volume
        self.catchment_area = catchment_area


    def __repr__(self):
        return "Check dam no %s" %(self.name)


