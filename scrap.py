__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import operator

import mpl_toolkits.mplot3d.axes3d

from numpy import pi, arange, sin, linspace

from bokeh.models import LinearAxis, Range1d
from bokeh.plotting import figure, show, output_file


# t = np.linspace(0,10,40)
#
# y = np.sin(t)
# z = np.sin(t)
# print t
# print y
# length = np.sqrt(y**2 + z **2)
# print length
# ax1 = plt.subplot(111,projection='3d')
# line, = ax1.plot(t,y,z,color='r',lw=2)
# arrow_1 = ax1.plot(t[0:2]*1.5, length[0:2], z[0:2], lw=3)
#
# plt.show()
x = arange(-2*pi, 2*pi, 0.1)
y = sin(x)
y2 = linspace(0, 100, len(x))

p = figure(x_range=(-6.5, 6.5), y_range=(-1.1, 1.1), min_border=80)

p.circle(x, y, fill_color="red", size=5, line_color="black")

p.extra_y_ranges['foo'] = Range1d(0, 100)
p.circle(x, y2, fill_color="blue", size=5, line_color="black", y_range_name="foo")
p.add_layout(LinearAxis(y_range_name="foo"), 'left')

output_file("twin_axis.html", title="twin_axis.py example")

show(p)
