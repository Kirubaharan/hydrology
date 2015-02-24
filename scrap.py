from __future__ import division
from fillplots import Plotter,boundary
import Pysolar as ps
import math
from datetime import timedelta
import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.interpolate import griddata
import os
from matplotlib.collections import PolyCollection
import brewer2mpl
from matplotlib.colors import colorConverter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors
from scipy.odr import *
from scipy import power
from matplotlib import cm as CM
from matplotlib.colors import from_levels_and_colors
from scipy.special import *
import matplotlib.patches as mpatches
from datetime import datetime
import matplotlib.dates as mdates
from bisect import bisect_left

# a = [[1,2], [2,3], [3,4], [4, 5], [1, 6], [2,7], [1,8]]
# df = pd.DataFrame(a,columns=['b','c'])
# z = df.groupby(['b']).apply(lambda tdf:pd.Series(dict([[vv,tdf[vv].unique().tolist()] for vv in tdf if vv not in ['b']])))
# z = df.groupby(['b']).apply(lambda tdf:pd.Series(dict([[vv,tdf[vv].unique().tolist()] for vv in tdf ])))
# z['d'] = 0.00
# z[['d']] = z[['d']].astype(float)
# for index in z.index:
#     list_c = z['c'][index]
#     z['d'][index] = (len(list_c))
#
# z['e'] = 1/z['d']
# z = z[['c', 'e']]
# print z
#
# print z['c'][1]
# print z['c'][2]
# print z['c'][3]
# print z['c'][4]
#
# xi = np.array([0., 0.5, 1.0])
# yi = np.array([0., 0.5, 1.0])
# zi = np.array([[0., 1.0, 2.0],[0., 1.0, 2.0],[-0.1, 1.0, 2.0]])
# v = np.linspace(0.1, 1.5, 10, endpoint=True)
# tick = v.flatten()
# tick = np.insert(tick,[0,10],[-0.1, 2.0])
# # levels = [-0.1, 0, 0.5, 1.0, 1.5, 2.0]
# CS = plt.contourf(xi, yi, zi, len(tick), cmap=plt.cm.jet, levels=tick)
# C = plt.contour(xi, yi, zi, len(tick), colors='black', levels=tick )
# plt.colorbar(CS, shrink=0.5, aspect=5, ticks= tick)
# plt.show()

# a = [[613, 42023, 20], [762, 32557, 20], [323, 63163,20], [643, 60877, 20],
#      [422, 32740, 20], [394,48891, 20], [744, 105341,20], [334, 58036,20],
#      [303,27343,20], [861,33608,20]]
# df = pd.DataFrame(a, columns=['a', 'b', 'kseq'])
# X = df.a
# Y = df.b
# Z = df.kseq
#
# fig = plt.figure(figsize=plt.figaspect(0.5), facecolor='grey')
# ax = fig.gca(projection='3d', axisbg='grey')
#
# # ax = fig.add_subplot(1, 2, 1, projection='3d')
# xi = np.linspace(X.min(), X.max(), 100)
# yi = np.linspace(Y.min(), Y.max(), 100)
# zi = griddata((X, Y), Z, (xi[None, :], yi[:, None]), method='linear')    # create a uniform spaced grid
# xig, yig = np.meshgrid(xi, yi)
# surf = ax.plot_surface(xig, yig, zi, rstride=5, cstride=3, linewidth=0, cmap=plt.cm.coolwarm, antialiased=False)   # 3d plot
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()


# def sph2cart(theta, phi, radius):
#     x_cart = radius * np.sin(theta) * np.cos(phi)
#     y_cart = radius * np.sin(theta) * np.sin(phi)
#     z_cart = radius * np.cos(theta)
#     return np.array([x_cart, y_cart, z_cart])
#
#
# def antenna_plot_3D(mod_F, T, P):
#     [x, y, z] = sph2cart(T, P, mod_F)
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1, projection='3d')
#     p_surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,
#                              linewidth=0, antialiased=True,
#                              cmap=plt.cm.jet)
#     ax.set_zlabel('Z axis', fontsize=15)
#     ax.set_xlabel('X axis', fontsize=15)
#     ax.set_ylabel('Y axis', fontsize=15)
#     fig.colorbar(p_surf, shrink=0.5, aspect=5)
#     # plt.savefig('3d_pattern.png')
#     plt.show()
#
# x = 300
# t = np.linspace(0, np.pi, num=x)
# p = np.linspace(0,2*np.pi, num=x)
# [T,P] = np.meshgrid(t,p)
# mod_F = np.sin(T) * np.cos(P)
# antenna_plot_3D(mod_F, T, P)
# plt.show()
# data = {'2014-11-15':1, '2014-11-16':2, '2014-11-17':3, '2014-11-18':5, '2014-11-19':8, '2014-11-20': 19}
# df = pd.DataFrame(list(data.iteritems()), columns=['Date', 'val'])
# df = df.set_index(pd.to_datetime(df.Date, format='%Y-%m-%d'))
# # print(df)
# o_list = []
# x_list = []
# check_list = [3,8,19]
# for index in df.index:
#     if df.val[index] in check_list:
#         o_list.append(index)
#     else:
#         x_list.append(index)
#
# df_o = df.ix[o_list]
# df_x = df.ix[x_list]
#
# fig = plt.figure()
# plt.plot_date(df_o.index, df_o.val, 'bo')
# plt.plot_date(df_x.index, df_x.val, 'bx')
# plt.show()
# df_list =[]
# file_list = []
# path = 'file_path'
# for file in file_list:
#     df_name = 'df_%s' %file
#     df_list.append(df_name)
#     ('df_%s' % file) = pd.read_csv(path+file)
#
#
# # new_df = pd.concat(df_list)
# import numpy as np
# import matplotlib.pyplot as plt
#
# xi = np.array([0., 0.5, 1.0])
# yi = np.array([0., 0.5, 1.0])
# zi = np.array([[0., 1.0, 2.0],
#                [0., 1.0, 2.0],
#                [-0.1, 1.0, 2.0]])
#contour levels
# print v
#
# print tick
# cmap = plt.cm.jet
# fig,ax = plt.subplots()
# C = ax.contour(xi, yi, zi, v, linewidths=0.5, colors='k')
# CS = ax.contourf(xi, yi, zi, v, cmap=plt.cm.jet, extend='both', origin='lower')
# cbarticks=np.hstack([zi.min(),v,zi.max()])#contour tick labels
# cbar=fig.colorbar(CS,extendrect='True',extendfrac='auto', spacing='proportional')
# print CS.cvalues
# # plt.show()
# import pandas as pd
# a = [[1,2], [2,3], [3,4], [4, 5], [1, 6], [2,7], [1,8]]
# df = pd.DataFrame(a,columns=['b','c'])
# print df
# z = df.groupby(['b']).apply(lambda tdf:pd.Series(dict([[vv,tdf[vv].unique().tolist()] for vv in tdf if vv not in ['b']])))
# print z
# z = z.sort_index()
# # print z['c'][1]
# # print z['c'][2]
# # print z['c'][3]
# # print z['c'][4]
# z['d'] = 0.000
# z[['d']] = z[['d']].astype(float)
# len_b = len(z.index)
# z['d'] = float(len_b)
# z['e'] = 1/z['d']
# z = z[['c', 'e']]
# # z.to_csv('your output folder')
# print z


# def sph2cart(theta, phi, radius):
#     x_cart = radius * np.sin(theta) * np.cos(phi)
#     y_cart = radius * np.sin(theta) * np.sin(phi)
#     z_cart = radius * np.cos(theta)
#     return np.array([x_cart, y_cart, z_cart])
#
# def antenna_plot_3D(mod_F, T, P):
#     [x, y, z] = sph2cart(T, P, mod_F)
#
#     colortuple = ('y', 'b', 'r', 'g' )
#     colors =np.empty(x.shape, dtype=str)
#     # for yi in range(len(y)):
#     #     for xi in range(len(x)):
#     #         if
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1, projection='3d')
#     p_surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,
#                              linewidth=0, antialiased=True,
#                              facecolors=plt.cm.jet)
#     ax.set_zlabel('Z axis', fontsize=15)
#     ax.set_xlabel('X axis', fontsize=15)
#     ax.set_ylabel('Y axis', fontsize=15)
#     fig.colorbar(p_surf)
#     # plt.savefig('3d_pattern.png')
#     # plt.show()
#
# x = 5
# t = np.linspace(0, np.pi, num=x)
# # print t
# p = np.linspace(0,2*np.pi, num=x)
# # print(p)
# T, P = np.meshgrid(t,p)
# # print T
# # print P
# mod_F = np.sin(T) * np.cos(P)
# # print mod_F
# # antenna_plot_3D(mod_F, T, P)
# x, y, z = sph2cart(T, P, mod_F)
# print 'x '
# print x
# # print 'y'
# # print y
# b = np.hstack((x, y))
# print 'b'
# # # print b
# ar = np.array([80, 64, 82, 72,  9, 35, 94, 58, 19, 41, 42, 18, 29, 46, 60, 14, 38,
#        19, 20, 34, 59, 64, 46, 39, 24, 36, 86, 64, 39, 15, 76, 93, 54, 14,
#        52, 25, 14,  4, 51, 55, 16, 32, 14, 46, 41, 40,  1,  2, 84, 61, 13,
#        26, 60, 76, 22, 77, 50,  7, 83,  4, 42, 71, 23, 56, 41, 35, 37, 86,
#         3, 95, 76, 37, 40, 53, 36, 24, 97, 89, 58, 63, 69, 24, 23, 95,  7,
#        55, 33, 42, 54, 92, 87, 37, 99, 71, 53, 71, 79, 15, 52, 37])
#
# ar[::-1].sort()
# y = np.cumsum(ar).astype("float32")
#
# # Normalize to a percentage
# y/=y.max()
# y*=100.
#
# # Prepend a 0 to y as zero stores have zero items
# y = np.hstack((0,y))
#
# # Get cumulative percentage of stores
# x = np.linspace(0,100,y.size)
#
# # Plot the normalized chart (the one on the right)
# f, ax = plt.subplots(figsize=(3,3))
# ax.plot(x,y)
# ax1 = ax.twinx().twiny()
# ax1.plot(x*len(ar), y*np.sum(ar), 'r')
# # format_figure(f,ax)
#
# # Plot the unnormalized chart (the one on the left)
# # f, ax = plt.subplots(figsize=(3,3))
# # ax.plot(x*len(ar), y*np.sum(ar))
# # # format_figure(f,ax)
# plt.show()
#
# x,y,c,s = np.random.rand(100), np.random.rand(100), np.random.rand(100)*100, np.random.rand(100)*100
# b = plt.scatter(x,y,c=c,s=s,cmap='YlGnBu', alpha=0.3)
# cbar = plt.colorbar(b)
# # c = cbar.ax.get_yticklabels()
# # label_list = []
# # for i in cbar.ax.get_yticklabels():
# #     a = int(i.get_text())
# #     label_list.append(a)
#
# cbar.set_ticks(ticks=[15,25,35,45,55,65,75,85,95],update_ticks=True)
# cbar.set_ticklabels(ticklabels=[10,20,30,40,60,70,80,90], update_ticks=True)
# # print label_list
# plt.show()
# raise SystemExit(0)
# XdS=[14.54156005,  14.53922242,  14.53688586,  14.53454823, 14.5322106 ,  14.52987297, 14.52753426,  14.52519555, 14.52285792,  14.52051922,  14.51818051, 14.51584073, 14.51350095, 14.51116117, 14.5088214 , 14.50648162, 14.50414076,  14.50179991,  14.49945906,  14.49711821]
# YdS=[31.13035144,  31.12920087,  31.12805245,  31.12690188, 31.12575131,  31.12460073,  31.12345016,  31.12229745, 31.12114473,  31.11999201,  31.1188393 , 31.11768443, 31.11652957,  31.11537471, 31.11421984, 31.11306283, 31.11190582,  31.11074882,  31.10959181,  31.1084348]
# ZdS=[3.94109446,  3.94060316,  3.94011186,  3.93962083,  3.93912926, 3.93863796,  3.93814639,  3.93765482,  3.93716325,  3.93667169, 3.93617985,  3.93568828,  3.93519618,  3.93470434,  3.9342125 , 3.9337204 ,  3.93322829,  3.93273592,  3.93224382,  3.93175144]
#
# xmin = min(XdS)
# ymin = max(YdS)
# zmin = min(ZdS)
# length_of_array = len(XdS)
# xmin_array = [xmin] * length_of_array
# ymin_array = [ymin] * length_of_array
# zmin_array = [zmin] * length_of_array
# fig=plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(XdS,YdS,ZdS,zdir='z', c='r')
# ax.plot(XdS,YdS,zmin_array, zdir='z', c='g')
# ax.plot(xmin_array, YdS, ZdS, 'y')
# ax.plot(XdS,ymin_array,ZdS,'b')
# # ax.plot(XdS, ymin, ZdS,zdir='z', c='y')
#
#
# ax.set_xlabel('XKSM (Saturn Radii)')
# ax.set_ylabel('YKSM (Saturn Radii)')
# ax.set_zlabel('ZKSM (Saturn Radii)')
# plt.show()
def gauss(x,y,Sigma,mu):
    X=np.vstack((x,y)).T
    mat_multi=np.dot((X-mu[None,...]).dot(np.linalg.inv(Sigma)),(X-mu[None,...]).T)
    return  np.diag(np.exp(-1*(mat_multi)))
# def plot_countour(x,y,z):
#     # define grid.
#     xi = np.linspace(-2.1,2.1,100)
#     yi = np.linspace(-2.1,2.1,100)
#     ## grid the data.
#     zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
#     levels = [0.2, 0.4, 0.6, 0.8, 1.0]
#     # contour the gridded data, plotting dots at the randomly spaced data points.
#     CS = plt.contour(xi,yi,zi,len(levels),linewidths=0.5,colors='k', levels=levels)
#     #CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
#     CS = plt.contourf(xi,yi,zi,len(levels),cmap=cm.Greys_r, levels=levels)
#     plt.colorbar() # draw colorbar
#     # plot data points.
#     # plt.scatter(x,y,marker='o',c='b',s=5)
#     plt.xlim(-2,2)
#     plt.ylim(-2,2)
#     plt.title('griddata test (%d points)' % npts)
#     plt.show()
#
#
# # make up some randomly distributed data
# seed(1234)
# npts = 1000
# x = uniform(-2,2,npts)
# y = uniform(-2,2,npts)
# z = gauss(x,y,Sigma=np.asarray([[1.,.5],[0.5,1.]]),mu=np.asarray([0.,0.]))
# plot_countour(x,y,z)
# start = 1406507532491431
# end = 1406535228420914

# start_ts = pd.to_datetime(start, unit='us') # Timestamp('2014-07-28 00:32:12.491431')
# end_ts = pd.to_datetime(end, unit='us')
# print start_ts
# print end_ts
# new_index = pd.date_range(start=start_ts.strftime('%Y-%m-%d %H:%M'), end=end_ts.strftime('%Y-%m-%d %H:%M'), freq='1min')
# print new_index
# for i in new_index:
#     print i
# fig, ax = plt.subplots()
# a = [[1,2], [2,3], [3,4], [4, 5], [1, 6], [2,7], [1,8]]
# df = pd.DataFrame(a,columns=['askdabndksbdkl','aooweoiowiaaiwi'])
# axs = pd.scatter_matrix( df, alpha=0.2, diagonal='kde')
# x_rotation = 90
# y_rotation = 90
# for row in axs:
#     for subplot in row:
#         setp(subplot.get_xticklabels(), rotation=x_rotation)
#         setp(subplot.get_yticklabels(), rotation=y_rotation)
# n = len(df.columns)
# for x in range(n):
#     for y in range(n):
#         ax = axs[x, y]
#         ax.xaxis.label.set_rotation(90)
#         ax.yaxis.label.set_rotation(0)
#         ax.yaxis.labelpad = 50
# ax = axs[1, 0]
# ax.xaxis.set_rotate_label(False)
# ax.xaxis.label.set_rotation(90)
# ax.yaxis.label.set_rotation(0)
# plt.show()


# df = pd.DataFrame.from_items([("A\tbar", [1, 2, 3]), ("B\tfoo" , [4, 5, 6])],orient='index', columns=['one', 'two', 'three'])
# df['col_a'] = df.index
# lista = [item.split('\t')[0] for item in df['col_a']]
# listb = [item.split('\t')[1] for item in df['col_a']]
# df['col_a'] = lista
# df['col_b'] = listb
# cols = df.columns.tolist()
# # cols = cols[-2:] + cols[:-2]
# # df = df[cols]
# print df

#
# data_time=pd.date_range(start="2011-01-01", periods=100, freq='D')
# s1 = pd.Series(np.random.randint(80,100,100))
# s2 = pd.Series(np.random.randint(60,70,100))
# x = pd.concat([s1,s2], axis=1)
# x.set_index(data_time, inplace=True)
# x.columns = ['count','registered']
#
# f,(ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
# ax1.plot(data_time.to_pydatetime(), x['count'])
# ax1.set_title('Date vs. Count')
# percentage = x['registered']/x['count']
# ax2.plot(data_time.to_pydatetime(), percentage)
# ax2.set_title('Date vs. Percentage of Registration')
# f.autofmt_xdate(rotation=90)
# plt.show()
#
# xpos = np.arange(0,4,1)
# ypos = np.arange(0,4,1)
# xpos, ypos = np.meshgrid(xpos, ypos)
# xpos = xpos.flatten()
# ypos = ypos.flatten()
# zpos = np.zeros(4*4)
# rho = np.random.random((4,4))
# dx = 0.5 * np.ones_like(zpos)
# dy = dx.copy()
# dz = rho.flatten()
# nrm=mpl.colors.Normalize(-1,1)
# colors=cm.RdBu(nrm(-dz))
# alpha = np.linspace(0.2, 0.95, 16, endpoint=True)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# for i in range(len(xpos)):
#     ax.bar3d(xpos[i],ypos[i],zpos[i], dx[i], dy[i], dz[i], alpha=alpha[i], color=colors[i], linewidth=0)
# plt.show()
# fig,ax1 = plt.subplots()
# a = np.random.normal(loc=np.random.rand(),size=100)
# b = np.random.normal(loc=10*np.random.rand(),size=100)
# print min(a)
# print max(a)
# print min(b)
# print max(b)
# frac_a = (max(a) - min(a))/10
# frac_b = (max(b) - min(b))/10
# ax1.plot(a,'r-')
# ax2 = ax1.twinx()
# ax2.plot(b,'b-')
# ax2.set_ylim(0,max(b)+frac_b)
# ax1.set_ylim(min(a)+frac_a,0)

# ts = ['20140101', '20140102', '20140105', '20140106', '20140107']
# xs = pd.Series(data=range(len(ts)), index=pd.to_datetime(ts))
# fig, ax = plt.subplots()
# xs.plot(use_index=False)
# a= ax.get_xticks()
# print a
# ax.set_xticklabels(pd.to_datetime(ts))
# ax.set_xticks(range(len(ts)))
# fig.autofmt_xdate()
# plt.show()
# results_bbias = OrderedDict()
# results_boutliers = OrderedDict()
# results_bscatters = OrderedDict()
# oddlist=[ 0.0, 0.5, 0.7, 0.8, 0.9, 0.95]
# for i in range(len(oddlist)):
#     results_bbias.update({'bias_%s'%str(oddlist[i]): numpy.random.random((5,3))})
#     results_boutliers.update({'outliers_%s'%str(oddlist[i]):numpy.random.random((1,3))})
#     results_bscatters.update({'scatters_%s'%str(oddlist[i]):numpy.random.random((5,3))})
# print results_bbias['bias_0.0']
# boutliers_list = []
# for i in results_boutliers.values():
#     boutliers_list.append(i)
# print boutliers_list
# name, array_bbias = results_bbias.items()
# df_bbias=pd.DataFrame(results_bbias['bias_0.0'],index=oddlist[0:5],columns=['magnitude','value', 'error'])
# df_bscatters=pd.DataFrame(list(results_bscatters.values()),index=oddlist,columns=['magnitude','value','error'])
# df_boutliers=pd.DataFrame(results_boutliers,index=oddlist,columns=['magnitude','value','error'])
# print df_bbias.head()
# print df_bscatters.head()
# print df_boutliers.head()
# data_broad=pd.concat([df_bbias, df_bscatters, df_boutliers],axis=1)
# print data_broad.head()
# # data_broad.to_csv("/home/kiruba/Documents/test.csv", header=True)
# input_file = '/home/kiruba/Documents/test_1.csv'
# df = pd.read_csv(input_file)
# print df
# df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')
# print df
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
#
#
# x   = np.linspace(0,1E15,10)
# y   = 1E-15*x-2
#
#
#
# ax1.set_xlim(-0.05E15,1.1E15)
# ax1.set_ylim(-2.1, -0.7)
#
#
# ax1.plot(x, y, 'o')
#
# # Fit using odr
# def f(B, x):
#     return B[0]*x + B[1]
# sx = np.std(x)
# sy = np.std(y)
# linear = Model(f)
# mydata = RealData(x=x,y=y, sx=sx, sy=sy)
# myodr = ODR(mydata, linear, beta0=[1.00000000e-15, 2.])
# myoutput = myodr.run()
# myoutput.pprint()
#
# a, b = myoutput.beta
# sa, sb = myoutput.sd_beta
#
# xp = np.linspace(min(x), max(x), 1000)
# yp = a*xp+b
# ax1.plot(xp,yp)
# plt.show()
# popt = np.polyfit(x,y,1)
# print popt
# fig = plt.figure()
#
# # create an example histogram which is asymmetrical around zero
# x = np.random.rand(400)
# y = np.random.rand(400)
# Z, xedges, yedges = np.histogram2d(x, y, bins=10)
# Z = Z - 2.
# #  -1 0 3 6 9
# cmap, norm = from_levels_and_colors([-1, 0, 3, 6, 9, 12], ['r', 'b', 'g', 'y', 'm'])
# plt.pcolormesh(xedges, yedges, Z, cmap=cmap, norm=norm)
# plt.colorbar()
# plt.show()

#
# q = 6.0/1000
# rhof = 1000
# lameu = 11.2*10**9
# lame = 8.4*10**9
# np.pi
# alpha = 0.65
# G = 8.4*10**9
# k = 1.0*10**(-15)
# eta = 0.001
# t = 1000*24*3600
#
# kappa = k/eta
# print "kappa ist:",kappa
# c = (kappa*(lameu-lame)*(lame+2*G))/((alpha**2)*(lameu+2*G))
# print "c ist:",c
#
#
# xmin = -100
# xmax = 100
# ymin = -100
# ymax = 100
#
# x = np.arange(xmin,xmax,5.0)
# y = np.arange(ymin,ymax,5.0)
# x, y = np.meshgrid(x, y)  # Erzeugung einer Matrix
#
#
#
# r = np.sqrt(x**2+y**2)
# P = (q/(rhof*4*np.pi*kappa))*(expn(1,(r**2)/(4*c*t)))
# z = P/1e6
# print z
# z[z==np.inf] = np.nan
# a = 0
# b= 0
# plt.figure()
# CS = plt.contour(x, y, z)
# points = plt.plot(a,b, 'ro')
# plt.xticks(np.arange(-80,80,5))
# plt.yticks(np.arange(-80,80, 5))
# plt.clabel(CS, inline=1, fontsize=10)
# plt.show()


# fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
# dep_color = {'a':'red','b':'blue','c':'green'}
# label = []
# for dep in dep_color:
#     P.append(mpatches.Patch(color=dep_color[dep], label=dep))
#     label.append(dep)
# label = tuple(label)
# fig.legend(handles=P,labels = label,bbox_to_anchor=(0.,1.02,1.,.102), loc=3,ncol=3, mode="expand", borderaxespad=0.)
# plt.show()pandas plot subplots two plotsh
# df = pd.read_csv('/home/kiruba/Documents/test/test_2.CSV', header=None, names=[ 'a', 'b', 'c', 'd'] )
# fig, ax = plt.subplots()
# df.plot(kind='barh', stacked=True, ax=ax)
# ax.set_yticklabels(['A', 'B', 'C', 'D', 'E', 'F'])
# time = pd.date_range(start=pd.to_datetime('07:00', format='%H:%M'), end=pd.to_datetime('13:00', format='%H:%M'),freq='H')
# time_x = [dt.strftime('%H:%M') for dt in time]
# ax.set_xticklabels(time_x)
# fig.autofmt_xdate()
# plt.show()
# print df
# y_cal = [10, 40, 100, 150, 200, 300]
# x_cal = [1971, 2336, 3083, 3720, 4335, 5604]
#
#
# def find_range(array, ab):
#     if ab < max(array):
#         start = bisect_left(array, ab)
#         return array[start-1], array[start]
#     else:
#         return min(array), max(array)

x = [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
y = [1.96, 1.97, 1.98, 2.0, 2.0, 1.99, 1.98, 1.98, 1.99, 1.98, 1.98, 1.96, 1.96, 1.96, 1.98]
print len(x)
print len(y)
fig = plt.figure()
plt.plot(x, y, '-bo')
plt.show()