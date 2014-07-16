
__author__ = 'kirubaharan'

##area of curve


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spread import spread
# copy the code from http://code.activestate.com/recipes/577878-generate-equally-spaced-floats/ #
import itertools
from matplotlib import rc

##read csv
csv_file = '/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/634_profile_3_sec.csv'
df = pd.read_csv(csv_file,header=0)

##plot
# curve1 = plt.plot(x,y_1)
# curve2 = plt.plot(x,y_2)
# curve3 = plt.plot(x,y_3)
# line1 = plt.plot([-8,7], [0.7,.7], lw=2)
# curve1 + curve2 + curve3 + line1
# plt.show()
## function to create pairs of iterable elevations
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2,s3), ..."
    a, b = itertools.tee(iterable)
    next(b,None)
    return itertools.izip(a, b)

#Enter the check dam height
check_dam_height = 0.70
#to create stage with 5 cm intervals
no_of_stage_interval = check_dam_height/.05
# to create series of stage values
dz = list((spread(0.00,check_dam_height,int(no_of_stage_interval), mode=3))) # dz = stage
# y=1
#empty list to store the results in the end
results_1 = []

# for every value of 5 cm iteration
for z in dz: 
#for every iteration initial value needs to be set to 0
    water_area_1 = 0 
# creates consecutive no, see above for function
    for y1, y2 in pairwise(df.Y1): 
#to find delta elevation
        delev = (y2 - y1) / 10
#assign initial value to elev
        elev = y1              
#for iterating over 10 cm strip, this creates no from 1 to 10
        for b in range(1,11,1): 
# finding the next elevation value after measured value
            elev = elev + delev 
# if water level is above the estimated elev,area needs to be determined otherwise 0            
            if  z > elev:  
                water_area_1 = water_area_1 + 0.1 * (z-elev)
# first section so dy = 1    
    calc_vol_1 = water_area_1 
# add the values to list    
    results_1.append(calc_vol_1)
##create pandas dataframe/array to store the values
index = [range(1,15,1)]
columns = ['stage_m']
data = np.array(dz)
output = pd.DataFrame(data,index=index,columns=columns)
#append results to dataframe
output['Volume_1'] = results_1 
# print output

# y=2
#empty list to store the results in the end/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/created_profile_607.csv
results_2 = []

# for every value of 5 cm iteration
for z in dz: 
#for every iteration initial value needs to be set to 0
    water_area_2 = 0 
# creates consecutive no, see above for function
    for y1, y2 in pairwise(df.Y2): 
#to find delta elevation
        delev = (y2 - y1) / 10
#assign initial value to elev
        elev = y1              
#for iterating over 10 cm strip, this creates no from 1 to 10
        for b in range(1,11,1): 
# finding the next elevation value after measured value
            elev = elev + delev 
# if water level is above the estimated elev,area needs to be determined otherwise 0            
            if  z > elev:  
                water_area_2 = water_area_2 + 0.1 * (z-elev)
# first section so dy = 1    
    calc_vol_2 = water_area_2 * 2
# add the values to list    
    results_2.append(calc_vol_2)
#append results to dataframe
output['Volume_2'] = results_2 
# print output

# y=1
#empty list to store the results in the end
results_3 = []

# for every value of 5 cm iteration
for z in dz: 
#for every iteration initial value needs to be set to 0
    water_area_3 = 0 
# creates consecutive no, see above for function
    for y1, y2 in pairwise(df.Y3):
#to find delta elevation
        delev = (y2 - y1) / 10
#assign initial value to elev
        elev = y1              
#for iterating over 10 cm strip, this creates no from 1 to 10
        for b in range(1,11,1): 
# finding the next elevation value after measured value
            elev = elev + delev 
# if water level is above the estimated elev,area needs to be determined otherwise 0            
            if  z > elev:  
                water_area_3 = water_area_3 + 0.1 * (z-elev)
# first section so dy = 1    
    calc_vol_3 = water_area_3 *2
# add the values to list    
    results_3.append(calc_vol_3)
#append results to dataframe
output['Volume_3'] = results_3 
# print output

# add all the corresponding values
output['total_volume'] = output['Volume_1']+ output['Volume_2']+ output['Volume_3']
print(output)
#plot values
plt.plot(output['stage_m'],output['total_volume'],label = "Stage - Volume")
plt.legend(loc = 'upper left')
# plt.xlabel('Stage (m)')
# plt.ylabel('Total Volume (cu.m')
# plt.title('Stage - volume relationship curve for Check Dam - 634')
##add axis labels
rc('font',**{'family':'sans-serif','sans-serif' : ['Helvetica']})
rc('text',usetex=True)
plt.rc('text',usetex=True)
plt.rc('font',family='serif')
plt.xlabel(r'\textbf{Stage} (m)')
plt.ylabel(r'\textbf{Volume} ($m^3$)')
plt.title(r"Stage - Volume Relationship for Check Dam 634",fontsize = 16)
plt.show()
# plt.savefig('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/stage_vol_634.png')
output.to_csv('/media/kiruba/New Volume/r/r_dir/stream_profile/new_code/test_634.csv',sep=",")