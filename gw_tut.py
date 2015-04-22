__author__ = 'kiruba'
import sys
import os
# import flopy.modflow as fmf
import flopy.modflow as fmf
import flopy.utils as fut
import numpy as np
import matplotlib.pyplot as plt
import fortranfile as ff
import flopy
from array import array
import struct
from scipy.interpolate import griddata
import pandas as pd
print flopy.__version__

# from numpy import array
# h = array([[4., 5., 6., 7.],
#           [4., 0., 0., 7,],
#           [4., 0., 0., 7.],
#           [4., 5., 6., 7.]])
# dummy = h.shape
# nrow = dummy[0]
# ncol = dummy[1]
#
# print 'Head matrix is a ', nrow, 'by', ncol, 'matrix'
#
# ni = 1
# conv_crit = 1e-3
# converged = False
# w = 1.1
#
# while (not converged):
#     # max_err = 0
#     for r in range(1, nrow-1):
#         for c in range(1, ncol-1):
#             h_old = h[r, c]
#             print h_old
#             h[r, c] = (h[r-1, c] + h[r+1, c] + h[r, c-1] + h[r, c+1])/4
#             print h[r, c]
#             c_1 = h[r, c] - h_old
#             print c_1
#             h[r, c] += (w * c_1)
#             print h[r, c]
#             diff = h[r, c] - h_old
#             print diff
#             if diff < conv_crit:
#                 converged = True
#     ni = ni +1
#
# # while (not converged):
# #     max_err = 0
# #     for r in range(1, nrow-1):
# #         for c in range(1, ncol-1):
# #             h_old = h[r, c]
# #             h[r, c] = (h[r-1, c] + h[r+1, c] + h[r, c-1] + h[r, c+1])/4
# #             diff = h[r, c] - h_old
# #             if (diff > max_err):
# #                 max_err = diff
# #     if (max_err < conv_crit):
# #         converged = True
# #     ni += 1
# print 'Number of iterations = ', ni-1
# print h



name = 'lake_example'
h1 = 100
h2 = 90
Nlay = 10
N = 101
L = 400.0
H = 50.0
k = 1.0

ml = fmf.Modflow(modelname=name, exe_name='/usr/local/src/mf2005/Unix/src/mf2005', version='mf2005', model_ws='mf_files/')



bot = np.linspace(-H/Nlay,-H,Nlay)
delrow = delcol = L/(N-1)
dis = fmf.ModflowDis(ml,nlay=Nlay,nrow=N,ncol=N,delr=delrow,delc=delcol,top=0.0,botm=bot,laycbd=0)

Nhalf = (N-1)/2
ibound = np.ones((Nlay,N,N))
ibound[:,0,:] = -1; ibound[:,-1,:] = -1; ibound[:,:,0] = -1; ibound[:,:,-1] = -1
ibound[0,Nhalf,Nhalf] = -1
start = h1 * np.ones((N,N))
start[Nhalf,Nhalf] = h2
bas = fmf.ModflowBas(ml,ibound=ibound,strt=start)
print "o"
lpf = fmf.ModflowLpf(ml, hk=k)
pcg = fmf.ModflowPcg(ml)
oc = fmf.ModflowOc(ml)
ml.write_input()
ml.run_model()

head_file = '/home/kiruba/PycharmProjects/area_of_curve/hydrology/hydrology/mf_files/lake_example.hds'
hds = fut.HeadFile(head_file)
h = hds.get_data(kstpkper=(1,1))
print len(h)
x = y = np.linspace(0, L, N)
c = plt.contour(x, y, h[0], np.arange(90,100.1,0.2))
plt.clabel(c, fmt='%2.1f')
plt.axis('scaled')
plt.show()
c = plt.contour(x,y,h[-1],np.arange(90,100.1,0.2))
plt.clabel(c,fmt='%1.1f')
plt.axis('scaled')
plt.show()
z = np.linspace(-H/Nlay/2,-H+H/Nlay/2,Nlay)
c = plt.contour(x,z,h[:,50,:],np.arange(90,100.1,.2))
plt.axis('scaled')
plt.show()

raise SystemExit(0)

infile = open('lake_example.hds', "rb")
blockdata = []
while infile.read((1)):
    infile.seek(-1,1)
    data = infile.read(56)
    n = struct.unpack('<3i4', data[0:12])
    n = struct.unpack('<2f4', data[12:20])
    n = struct.unpack('<5i4', data[36:56])
    ncol = n[0]
    nrow = n[1]
    a = np.fromfile(infile, dtype='f4', count=ncol*nrow).reshape((ncol, nrow))
    blockdata.append(a)
    data = infile.read(4)
    n = struct.unpack('<i4', data)

# for block in blockdata[0]:
#     print block
df = pd.DataFrame(blockdata[2])
# df.to_csv('heads_1.csv', sep=',')
# raise SystemExit(0)
x = y = np.linspace(0, L, N)
print len(x)
# print df
head = df.ix[1:9, 0:10]
print head
print len(head)
# print len(blockdata[0])
# print blockdata[0]
raise SystemExit(0)
xi = yi = np.linspace(0, L, 200)
# zi = griddata((x,y), blockdata[2][1:9, 1:9], (xi[None, :], yi[:, None]), method='linear')
# fig = plt.figure()
c = plt.contour(x, y, blockdata[2])
plt.show()
# # a = ff.FortranFile("lake_example.hds",mode='w')
# # a.writeReals(np.linspace(0,1,10))
# # a.close()
# # hds = fut.HeadFile('lake_example.hds')