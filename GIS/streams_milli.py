__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ftplib
import re
import os
import sys
import calendar
import datetime
from datetime import date
import gdal
from gdalconst import *
import progressbar
from pymodis.convertmodis_gdal import convertModisGDAL
import fnmatch
import gdalnumeric
import Image, ImageDraw
import ogr
import subprocess
import osr
import mpl_toolkits.basemap.pyproj as pyproj
import operator
from gdalnumeric import *
import ast
from shutil import copyfile
import pysal as ps



def getPRJwkt(epsg):
    '''Grab an WKT version of an EPSG code
       usage getPRJwkt(4326)
       This makes use of links like
       http://spatialreference.org/ref/epsg/4326/prettywkt/
       need internet to work'''
    import urllib
    f = urllib.urlopen("http://spatialreference.org/ref/epsg/{0}/prettywkt/".format(epsg))
    return(f.read())


def imageToArray(i):
    """
    Converts a  Python Imaging Library image to gdalnumeric array.
    """
    a = np.fromstring(i.tostring(),'b')
    print a.shape
    a.shape = i.im.size[1], i.im.size[0]
    return a


def arrayToImage(a):
    """
    Converts a gdalnumeric array to a
    Python Imaging Library Image.
    """
    i=Image.frombytes('L',(a.shape[1],a.shape[0]),
            (a.astype('b')).tostring())
    return i


def clip_raster_by_vector(input_folder, mask_shapefile, output_folder, file_extension='*.tif', t_srs='EPSG:32643', no_data=32767 ):
    files_list = os.listdir(input_folder)
    ds = ogr.Open(mask_shapefile)
    lyr = ds.GetLayer(0)
    lyr.ResetReading()
    ft = lyr.GetNextFeature()
    for item in files_list:
        if fnmatch.fnmatch(item, file_extension):
            in_raster = input_folder + '/' + item
            out_raster = output_folder + '/' +'tg_' + item
            subprocess.call(['gdalwarp', in_raster, out_raster, '-cutline', mask_shapefile, '-t_srs', t_srs, '-crop_to_cutline', '-dstnodata', "%s" %no_data])



def dbf2df(dbf_path, index=None, cols=False, incl_index=False):
    '''
    Read a dbf file as a pandas.DataFrame, optionally selecting the index
    variable and which columns are to be loaded.
    __author__  = "Dani Arribas-Bel <darribas@asu.edu> "
    https://github.com/GeoDaSandbox/sandbox
    ...
    Arguments
    ---------
    dbf_path    : str
                  Path to the DBF file to be read
    index       : str
                  Name of the column to be used as the index of the DataFrame
    cols        : list
                  List with the names of the columns to be read into the
                  DataFrame. Defaults to False, which reads the whole dbf
    incl_index  : Boolean
                  If True index is included in the DataFrame as a
                  column too. Defaults to False
    Returns
    -------
    df          : DataFrame
                  pandas.DataFrame object created
    '''
    db = ps.open(dbf_path)
    if cols:
        if incl_index:
            cols.append(index)
        vars_to_read = cols
    else:
        vars_to_read = db.header
    data = dict([(var, db.by_col(var)) for var in vars_to_read])
    if index:
        index = db.by_col(index)
        db.close()
        return pd.DataFrame(data, index=index)
    else:
        db.close()
        return pd.DataFrame(data)

tg_raster_file = '/media/kiruba/New Volume/milli_watershed/LISSIV_Georectified_images/26FEB2014/FEB_26_2014_3bands_tgh.img'
tg_raster = gdal.Open(tg_raster_file, GA_ReadOnly)
tg_infra_array = np.array(tg_raster.GetRasterBand(3).ReadAsArray())
xmin = np.min(tg_infra_array)
xmax = np.max(tg_infra_array)
y, x = np.histogram(tg_infra_array, bins=np.linspace(xmin, xmax, (xmax-xmin)/20))
nbins = y.size
print np.min(tg_infra_array)
print np.max(tg_infra_array)
dbf_file = '/media/kiruba/New Volume/milli_watershed/stream_survey/infra_red_feb_liss_4_stream_1.dbf'
streams_had_ir = dbf2df(dbf_file)
print streams_had_ir.head()
fig = plt.figure()
# plt.bar(x[:-1], y, width=x[1]-x[0], color='red', alpha=0.5)
plt.hist(streams_had_ir['FEB_26_2_3'].values)
plt.show()

gisbase = os.environ['GISBASE'] = ""

