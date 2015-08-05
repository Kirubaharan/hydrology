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
# gdal.UseExceptions()


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
# output_folder = "/media/kiruba/New Volume/MODIS/ET/scratch"
# convert modis to geotiff
# subset
# img = gdal.Open(output_folder+'/'+"MOD16A2.A2014M01.h25v07.105.2015035084256.hdf")
# subset = img.GetSubDatasets()
# print img.RasterXSize
# et_data = gdal.Open(subset[0][0])
# et = et_data.ReadAsArray()
# et_scaled = et*0.1
# plt.imshow(et_scaled, interpolation='nearest', vmin=10, cmap=plt.cm.gist_earth)
# plt.colorbar()
# plt.show()
# print subset[0][0]


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


def reproject_dataset(dataset, output_file_name, wkt_from="+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs", epsg_to=32643, pixel_spacing=1000, file_format='GTiff'):
    '''
    :param dataset: Modis subset name use gdal.GetSubdatasets()
    :param output_file_name: file location along with proper extension
    :param wkt_from: Modis wkt (default)
    :param epsg_to: integer default(32643)
    :param pixel_spacing: in metres
    :param file_format: default GTiff
    :return:
    '''
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(epsg_to)
    modis = osr.SpatialReference()
    modis.ImportFromProj4(wkt_from)
    tx = osr.CoordinateTransformation(modis, wgs84)
    g = gdal.Open(dataset)
    geo_t = g.GetGeoTransform()
    print geo_t
    x_size = g.RasterXSize
    y_size = g.RasterYSize
    (ulx, uly, ulz) = tx.TransformPoint(geo_t[0], geo_t[3])
    (lrx, lry, lrz) = tx.TransformPoint(geo_t[0] + (geo_t[1]*x_size), geo_t[3]+ (geo_t[5]*y_size))
    mem_drv = gdal.GetDriverByName(file_format)
    dest = mem_drv.Create(output_file_name, int((lrx-ulx)/pixel_spacing), int((uly - lry)/pixel_spacing), 1, gdal.GDT_Float32)
    new_geo = ([ulx, pixel_spacing, geo_t[2], uly, geo_t[4], -pixel_spacing])
    dest.SetGeoTransform(new_geo)
    dest.SetProjection(wgs84.ExportToWkt())
    gdal.ReprojectImage(g, dest, modis.ExportToWkt(), wgs84.ExportToWkt(), gdal.GRA_Bilinear)
    print "reprojected"





def reproject_modis_to_geotiff(input_folder, dest_prj=32643):
    '''
    Function to convert all modis hdf in folder to geotiff using gdal
    required libraries: osr, gdal
    :param input_folder: where hds files are stored
    :param dest_prj: default is wgs 84 / UTM 43N (EPSG 32643)
    '''
    files_list = os.listdir(input_folder)
    for item in files_list:
        if fnmatch.fnmatch(item, '*.hdf'):
            input_file_name = input_folder + '/' + item
            output_file_name = input_folder + '/' + item[0:23]
            img = gdal.Open(input_file_name)
            subset = img.GetSubDatasets()
            in_raster = subset[0][0]
            reproject_dataset(dataset=in_raster, output_file_name=output_file_name+ '.tif')

# reproject
# reproject_modis_to_geotiff(input_folder="/media/kiruba/New Volume/MODIS/ET/scratch")

# This function will convert the rasterized clipper shapefile
# to a mask for use within GDAL.

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

# clip the raster only for tg halli area
# http://geospatialpython.com/2011/02/clip-raster-using-shapefile.html


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


input_folder = "/media/kiruba/New Volume/MODIS/ET/scratch"
output_folder = "/media/kiruba/New Volume/MODIS/ET/scratch/TG_halli"
in_shape = "/media/kiruba/New Volume/MODIS/TG_halli/TGHalliCatchment_utm.shp"

# clip_raster_by_vector(input_folder=input_folder,mask_shapefile=in_shape, output_folder=output_folder)

# save as geotiff
def process_modis(input_folder, output_folder, scale_factor=0.1, null_value=32760, file_extension='*.tif'):
    files_list = os.listdir(input_folder)
    for item in files_list:
        if fnmatch.fnmatch(item, file_extension):
            in_raster = input_folder + '/' + item
            in_raster = gdal.Open(in_raster, GA_ReadOnly)
            out_raster = output_folder + '/' + 'proc_' + item
            band1 = in_raster.GetRasterBand(1)
            in_array = BandReadAsArray(band1)
            in_array[in_array > null_value] = np.nan
            data_out = in_array * scale_factor
            gdalnumeric.SaveArray(data_out, filename=out_raster, format='GTiff', prototype=in_raster)

input_folder = "/media/kiruba/New Volume/MODIS/ET/scratch/TG_halli"
output_folder = "/media/kiruba/New Volume/MODIS/ET/scratch/TG_halli/processed"
# process_modis(input_folder=input_folder, output_folder=output_folder)

# remove builtup and water bodies from modis raster
# 4000 builtup
# 4001 lake
builtup_shape = '/media/kiruba/New Volume/MODIS/ET/scratch/TG_halli/processed/lake_builtup/KFT_TGHALLI_BUILTUP_GE_utm.shp'
lake_shape = '/media/kiruba/New Volume/MODIS/ET/scratch/TG_halli/processed/lake_builtup/Tghalli_Lakes_utm.shp'

# subprocess.call(['gdalwarp', in_raster, out_raster, '-cutline', mask_shapefile, '-t_srs', t_srs, '-crop_to_cutline', '-dstnodata', "%s" %no_data])

print subprocess.list2cmdline(['gdal_rasterize -b 1 -burn 4000 -l KFT_TGHALLI_BUILTUP_GE_utm', builtup_shape,  '/media/kiruba/New Volume/MODIS/ET/scratch/TG_halli/processed/lake_builtup/builtup.tif'])
print subprocess.list2cmdline(['gdal_rasterize -b 1 -burn 5000 -l Tghalli_Lakes_utm', lake_shape, '/media/kiruba/New Volume/MODIS/ET/scratch/TG_halli/processed/lake_builtup/lake.tif'])


subprocess.call(['gdal_rasterize -b 1 -burn 4000 -l KFT_TGHALLI_BUILTUP_GE_utm', builtup_shape,  '/media/kiruba/New Volume/MODIS/ET/scratch/TG_halli/processed/lake_builtup/builtup.tif'])
# subprocess.call(['gdal_rasterize -b 1 -burn 5000 -l Tghalli_Lakes_utm', lake_shape, '/media/kiruba/New Volume/MODIS/ET/scratch/TG_halli/processed/lake_builtup/lake.tif'])




def raster_histogram_plot(raster):
    plt.hist(raster, 20,histtype='bar')
    plt.show()


raster_file = '/media/kiruba/New Volume/MODIS/ET/scratch/TG_halli/processed/proc_tg_MOD16A2.A2014M02.h25v07.tif'
g_raster = gdal.Open(raster_file, GA_ReadOnly)
band_1 = g_raster.GetRasterBand(1)
band_1_array = BandReadAsArray(band_1)

band_1_array = band_1_array[np.isfinite(band_1_array)]
print np.min(band_1_array)
print np.max(band_1_array)
raster_histogram_plot(band_1_array)




