__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from scipy.optimize import curve_fit
import math

dry_wb_file = '/media/kiruba/New Volume/ACCUWA_Data/Checkdam_water_balance/591/dry_wb_check.CSV'
dry_wb_df = pd.read_csv(dry_wb_file)
print dry_wb_df.head()


def fitfunc(h, alpha, beta):
    return alpha*(h**beta)


def errfunc(alpha,beta , x, y):
    return y - fitfunc(x,alpha, beta)


x = dry_wb_df['stage(m)']
y = dry_wb_df['infiltration(cu.m)']
p = [13.160, 0.969]
popt, pcov = curve_fit(fitfunc, x, y,p0=p)
print popt
error =  errfunc(popt[0], popt[1], x, y)
print error
fig = plt.figure()
y_new = fitfunc(x,popt[0], popt[1])
plt.plot(x, y, 'bo')
plt.plot(x, y_new, 'r')
plt.errorbar(x, y_new,color='k', yerr=error)
plt.title("Stage vs Infiltration")
plt.xlabel("Stage(m)")
plt.ylabel("Infiltration (cu.m)")
plt.show()
log_x = np.log(x)
log_y = np.log(y)
print log_y
OK = log_y == log_y
masked_log_y = log_y[OK]
masked_log_x = log_x[OK]
pars = np.polyfit(masked_log_x, masked_log_y, 1)
print pars
fitted_y = np.polyval(pars,masked_log_x)
error_y = masked_log_y - fitted_y
fig = plt.figure()
plt.plot(masked_log_x, masked_log_y, 'bo')
plt.plot(masked_log_x, fitted_y, 'r-')
plt.errorbar(masked_log_x,fitted_y, yerr=error_y, marker='.', color='g')
plt.xlabel("ln(stage(m)")
plt.ylabel("ln(Infiltration (cu.m)")
plt.title("Log plot")
plt.show()
print pars[0]
print math.exp(pars[1])
print np.mean(error)
print " # Infiltration rate #"
x = dry_wb_df['stage(m)']
y = dry_wb_df['infiltration rate (m/day)']
p = [-0.1074, 0.011]
popt, pcov = curve_fit(fitfunc, x, y,p0=p)
print popt
error =  errfunc(popt[0], popt[1], x, y)
print error
y_new = fitfunc(x,popt[0], popt[1])
fig = plt.figure()
plt.plot(x, y, 'bo')
# plt.plot(x, y_new, 'r')
plt.errorbar(x, y_new,color='k', yerr=error)
plt.title("Stage vs Infiltration rate")
plt.xlabel("Stage(m)")
plt.ylabel("Infiltration rate (m/day)")
plt.show()
log_x = np.log(x)
log_y = np.log(y)
print log_y
OK = log_y == log_y
masked_log_y = log_y[OK]
masked_log_x = log_x[OK]
pars = np.polyfit(masked_log_x, masked_log_y, 1)
print pars
fitted_y = np.polyval(pars,masked_log_x)
error_y = masked_log_y - fitted_y
fig = plt.figure()
plt.plot(masked_log_x, masked_log_y, 'bo')
plt.plot(masked_log_x, fitted_y, 'r-')
plt.errorbar(masked_log_x,fitted_y, yerr=error_y, marker='.', color='g')
plt.title("Log plot")
plt.xlabel("ln(Stage(m))")
plt.ylabel("ln(Infiltration rate(m/day))")
plt.show()
print pars[0]
print math.exp(pars[1])
print np.mean(error)