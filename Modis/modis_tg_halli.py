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
import glob
import gdalnumeric
import Image
import ogr
import operator
import ImageDraw
import subprocess
import osr



# monthly et is from 2000-2014
save_path = "/media/kiruba/New Volume/ACCUWA_Data/MODIS/ET/"
units = "mm/month"


def download_hdf_month(month, year, tile,  output_folder):
    """
    :param month: M
    :type month: int
    :param year: YYYY
    :type year: str
    :param tile: hXXvYY
    :type tile: str
    :param output_folder:
    :type output_folder: str
    :return:
    """
    print "Finding file"
    ftp_addr = "ftp.ntsg.umt.edu"
    ftp = ftplib.FTP(ftp_addr)
    ftp.login()
    dir_path = "pub/MODIS/NTSG_Products/MOD16/MOD16A2_MONTHLY.MERRA_GMAO_1kmALB/Y" + year + "/M" + "{0:02d}".format(month) + "/"
    try:
        ftp.cwd(dir_path)
    except:
        print("[ERROR] No data for that date")
        sys.exit(0)
    try:
        files = ftp.nlst()
    except:
        print("[ERROR] Unable to access the FTP server")
        sys.exit(0)
    hdf_pattern = re.compile('MOD16A2.A'+year+'M' + "{0:02d}".format(month) + '.'+tile+'.105.*.hdf$', re.IGNORECASE)
    matched_file = ''
    for f in files:
        if re.match(hdf_pattern, f):
            matched_file = f
            break
    if matched_file == '':
        print("[ERROR] No data for that tile")
        sys.exit(0)
    print("Found: " + matched_file)
    print("Downloading File")
    save_file = open(output_folder + "/" + matched_file, 'wb')
    ftp.retrbinary("RETR " +    matched_file, save_file.write)
    save_file.close()
    ftp.close()

# for m in range(1, 3, 1):
#     download_hdf_month(month=m, year='2014', tile="h25v07" ,output_folder="/media/kiruba/New Volume/MODIS/ET/scratch")
# raise SystemExit(0)
output_folder = "/media/kiruba/New Volume/MODIS/ET/scratch"
# convert modis to geotiff
# subset
img = gdal.Open(output_folder+'/'+"MOD16A2.A2014M01.h25v07.105.2015035084256.hdf")
subset = img.GetSubDatasets()
print img.RasterXSize
et_data = gdal.Open(subset[0][0])
et = et_data.ReadAsArray()
et_scaled = et*0.1
plt.imshow(et_scaled, interpolation='nearest', vmin=10, cmap=plt.cm.gist_earth)
plt.colorbar()
plt.show()
print subset[0][0]


#projection information
def getPRJwkt(epsg):
    '''Grab an WKT version of an EPSG code
       usage getPRJwkt(4326)
       This makes use of links like
       http://spatialreference.org/ref/epsg/4326/prettywkt/
       need internet to work'''
    import urllib
    f = urllib.urlopen("http://spatialreference.org/ref/epsg/{0}/prettywkt/".format(epsg))
    return(f.read())


def reproject_dataset(dataset, wkt_from="+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs", epsg_to=4326):
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(epsg_to)
    modis = osr.SpatialReference()
    modis.ImportFromProj4(wkt_from)
    tx = osr.CoordinateTransformation(modis, wgs84)
    g = gdal.Open(dataset)
    geo_t = g.GetGeoTransform()
    x_size = g.RasterXSize()
    y_size = g.RasterYSize()
    (ulx, uly, ulz) = tx.TransformPoint(geo_t[0], geo_t[3])
    (lrx, lry, lrz) = tx.TransformPoint(geo_t[0] + geo_t[1]*x_size, geo_t[3]+ geo_t[5]*y_size)
    mem_drv = gdal.GetDriverByName('MEM')
    dest = mem_drv.Create('', int((lrx-ulx)/geo_t[1]), int((uly - lry)/geo_t[1]), 1, gdal.GDT_Float32)
    new_geo = (ulx, geo_t[1], geo_t[2], uly, geo_t[4], geo_t[5])
    res = gdal.ReprojectImage(g, dest, modis.ExportToWkt(), wgs84.ExportToWkt(), gdal.GRA_Bilinear)
    return dest




def reproject_modis_to_geotiff(input_folder, dest_prj=4326):
    '''
    Function to convert modis to geotiff using gdal
    required libraries: os, gdal,pymodis
    used import statements
    import os
    import gdal
    from pymodis.convertmodis_gdal import convertModisGDAL
    :param input_folder: where hds files are stored
    :param dest_prj: default is wgs 84 (EPSG 4326)
    :return: modis data in geotiff in wgs 84 projection
    '''
    files_list = os.listdir(input_folder)
    out_prj = osr.SpatialReference()
    out_prj.ImportFromEPSG(dest_prj)
    modis_wkt_from = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
    in_prj = osr.SpatialReference()
    in_prj.ImportFromProj4(modis_wkt_from)
    tx  = osr.CoordinateTransformation(in_prj, out_prj)
    for item in files_list:
        if fnmatch.fnmatch(item, '*.hdf'):
            input_file_name = input_folder + '/' + item
            output_file_name = input_folder + '/' + item[0:23]
            img = gdal.Open(input_file_name)
            subset = img.GetSubDatasets()
            in_raster = gdal.Open(subset[0][0])
            in_geotransform = in_raster.GetGeoTransform()
            print in_geotransform
            print in_geotransform[1]
            if in_geotransform != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
                x_size = in_raster.RasterXSize
                y_size = in_raster.RasterYSize
                print x_size, y_size
                (ulx, uly, ulz) = tx.TransformPoint(in_geotransform[0], in_geotransform[3])
                (lrx, lry, lrz) = tx.TransformPoint(in_geotransform[0] + in_geotransform[1]*x_size, in_geotransform[3] + in_geotransform[5]*y_size)
                mem_drv = gdal.GetDriverByName('MEM')
                dest = mem_drv.Create('', int((lrx-ulx)/in_geotransform[1]), int((uly-lry)/in_geotransform[1]), 1, gdal.GDT_Float32)
                print dest.RasterXSize
                new_geo = (ulx, in_geotransform[1], in_geotransform[2], uly, in_geotransform[4], in_geotransform[5])
                dest.SetGeoTransform(new_geo)
                dest.SetProjection(out_prj.ExportToWkt())
                res = gdal.ReprojectImage(in_raster, dest, in_prj, out_prj, gdal.GRA_Bilinear)
                return dest
            else:
                print "projection error"

(image1, image2) = reproject_modis_to_geotiff(input_folder="/media/kiruba/New Volume/MODIS/ET/scratch")


def plot_gdal_file ( input_dataset, vmin=0, vmax=100 ):
    #plt.figure ( figsize=(11.3*0.8, 8.7*0.8), dpi=600 ) # This is A4. Sort of

    geo = input_dataset.GetGeoTransform() # Need to get the geotransform (ie, corners)
    data = input_dataset.ReadAsArray()
    # We need to flip the raster upside down
    data = np.flipud( data )
    # Define a cylindrical projection
    projection_opts={'projection':'cyl','resolution':'l'}
    # These are the extents in the native raster coordinates
    extent = [ geo[0],  geo[0] + input_dataset.RasterXSize*geo[1], \
        geo[3], geo[3] + input_dataset.RasterYSize*geo[5]]
    print geo
    print extent
    print input_dataset.RasterXSize
    print input_dataset.RasterYSize
    map = Basemap( llcrnrlon=extent[0], llcrnrlat=extent[3], \
                 urcrnrlon=extent[1],urcrnrlat=extent[2],  ** projection_opts)



    cmap = plt.cm.gist_rainbow
    cmap.set_under ('0.8' )
    cmap.set_bad('0.8' )
    cmap.set_over ('0.8')

    map.imshow( data, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest')



    map.drawcoastlines (linewidth=0.5, color='k')
    map.drawcountries(linewidth=0.5, color='k')

    map.drawmeridians( np.arange(0,360,5), color='k')
    map.drawparallels( np.arange(-90,90,5), color='k')
    map.drawmapboundary()
    cb = plt.colorbar( orientation='horizontal', fraction=0.10, shrink=0.8)
    cb.set_label(r'$10\cdot LAI$')
    plt.title(r'LAI')

plot_gdal_file(image1)
plt.show()
# This function will convert the rasterized clipper shapefile
# to a mask for use within GDAL.
raise SystemExit(0)

def imageToArray(i):
    """
    Converts a Python Imaging Library array to a
    gdalnumeric image.
    """
    a=gdalnumeric.fromstring(i.tostring(),'b')
    a.shape=i.im.size[1], i.im.size[0]
    return a


def arrayToImage(a):
    """
    Converts a gdalnumeric array to a
    Python Imaging Library Image.
    """
    i=Image.frombytes('L',(a.shape[1],a.shape[0]),
            (a.astype('b')).tostring())
    return i

# multiply to get et, exclude other values
# http://geospatialpython.com/2011/02/clip-raster-using-shapefile.html
input_folder = "/media/kiruba/New Volume/MODIS/ET/scratch"
files_list = os.listdir(input_folder)
layer_name = "TGHalliCatchment"
output_folder = "/media/kiruba/New Volume/MODIS/ET/scratch/TG_halli"
in_shape = "/media/kiruba/New Volume/MODIS/TG_halli/TGHalliCatchment.shp"
ds = ogr.Open(in_shape)
lyr = ds.GetLayer(0)
lyr.ResetReading()
ft = lyr.GetNextFeature()

for item in files_list:
    if fnmatch.fnmatch(item, '*.tif'):
        print item
        in_raster = input_folder + '/' + item
        out_raster = output_folder + '/' +'tg_' + item
        print out_raster
        subprocess.call(['gdalwarp', in_raster, out_raster, '-cutline', in_shape, '-t_srs', 'EPSG:4326', '-crop_to_cutline', '-dstnodata', "32767"])




# reproject
# save as geotiff


