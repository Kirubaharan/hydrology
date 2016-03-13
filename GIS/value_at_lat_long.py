import os, sys, time, gdal
from gdalconst import *


# coordinates to get pixel values for
# xValues = [77.527267, 77.55827, 77.55606, 77.55880, 77.55807, 77.55935, 77.55905, 77.53556, 77.52778, 77.53615]
# yValues = [13.360354, 13.36420, 13.36411, 13.36586, 13.36838, 13.36648, 13.36562, 13.35314, 13.36007, 13.35405]
xValues = [77.5433]
yValues = [13.3556]
ch_623_lat = 13.353
ch_623_long = 77.549
# ch_599_lat = 13.250119
# ch_599_long = 77.514195
# ch_625_lat = 13.364112
# ch_625_long = 77.556057
# set directory
os.chdir(r'/media/kiruba/New Volume/ACCUWA_Data/DEM_20_May/arkavathy/DEM_arkavathy_aster_15m')
# file:///media/kiruba/New Volume/ACCUWA_Data/DEM_20_May/arkavathy/DEM_arkavathy_aster_15m/merged_dem
# register all of the drivers
gdal.AllRegister()
# open the image
ds = gdal.Open('merged_dem', GA_ReadOnly)

if ds is None:
    print 'Could not open image'
    sys.exit(1)

# get image size
rows = ds.RasterYSize
cols = ds.RasterXSize
bands = ds.RasterCount


# get georeference info
transform = ds.GetGeoTransform()
xOrigin = transform[0]
yOrigin = transform[3]
pixelWidth = transform[1]
pixelHeight = transform[5]

# loop through the coordinates
for xValue,yValue in zip(xValues,yValues):
    # get x,y
    x = xValue
    y = yValue

    # compute pixel offset
    xOffset = int((x - xOrigin) / pixelWidth)
    yOffset = int((y - yOrigin) / pixelHeight)
    # create a string to print out
    s = "%s %s %s %s " % (x, y, xOffset, yOffset)

    # loop through the bands
    band = ds.GetRasterBand(1) # 1-based index
    # read data and add the value to the string
    data = band.ReadAsArray(xOffset, yOffset, 1, 1)
    value = data[0,0]
    print value
    s = "%s%s " % (s, value)
    print value
    # print out the data string
    print s

    # figure out how long the script took to run