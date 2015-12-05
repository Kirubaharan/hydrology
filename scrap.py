__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import operator

import mpl_toolkits.mplot3d.axes3d



t = np.linspace(0,10,40)

y = np.sin(t)
z = np.sin(t)
print t
print y
length = np.sqrt(y**2 + z **2)
print length
ax1 = plt.subplot(111,projection='3d')
line, = ax1.plot(t,y,z,color='r',lw=2)
arrow_1 = ax1.plot(t[0:2]*1.5, length[0:2], z[0:2], lw=3)

plt.show()