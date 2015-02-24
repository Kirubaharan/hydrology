__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from scipy.optimize import curve_fit
import math

base_file = "/media/kiruba/New Volume/water_quality/spss.csv"
wq_df = pd.read_csv(base_file, header=0)
wq_df.Time = wq_df.Time/1.0
print wq_df.head()

def func(t, cf, c, kf):
    return (math.log(cf)+ c)/(-1.0*kf*t)
#
popt, pcov = curve_fit(func, wq_df.Time, wq_df.Fraction)
print(popt)
# print pcov
fig = plt.figure()
plt.plot(wq_df.Time, wq_df.Fraction, 'ro')
plt.show()