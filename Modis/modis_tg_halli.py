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

# monthly et is from 2000-2014
year_string = "2000"  # YYYY
tile_name = "h25v07"   #hXXvYY
save_path = "/media/kiruba/New Volume/ACCUWA_Data/MODIS/ET/"
area = "/media/kiruba/New Volume/MODIS/TG_halli/TGHalliCatchment.shp"
units = "mm/month"

#Download hdf
# for m in range(1, 2, 1):
#     print m
#     print "Finding file"
#     ftp_addr = "ftp.ntsg.umt.edu"
#     ftp = ftplib.FTP(ftp_addr)
#     ftp.login()
#     dir_path = "pub/MODIS/NTSG_Products/MOD16/MOD16A2_MONTHLY.MERRA_GMAO_1kmALB/Y" + year_string + "/M" + "{0:02d}".format(m) + "/"
#     print dir_path
#     try:
#         ftp.cwd(dir_path)
#     except:
#         print("[ERROR] No data for that date")
#         sys.exit(0)
#     try:
#         files = ftp.nlst()
#     except:
#         print("[ERROR] Unable to access the FTP server")
#         sys.exit(0)
#     hdf_pattern = re.compile('MOD16A2.A'+year_string+'M' + "{0:02d}".format(m) + '.'+tile_name+'.105.*.hdf$', re.IGNORECASE)
#     matched_file = ''
#     for f in files:
#         if re.match(hdf_pattern, f):
#             matched_file = f
#             break
#     if matched_file == '':
#         print("[ERROR] No data for that tile")
#         sys.exit(0)
#     print("Found: " + matched_file)
#     print("Downloading File")
#     L = "/media/kiruba/New Volume/MODIS/ET/scratch"
#     save_file = open(L + "/" + matched_file, 'wb')
#     ftp.retrbinary("RETR " + matched_file, save_file.write)
#     save_file.close()
#     ftp.close()


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

for m in range(1, 2, 1):
    download_hdf_month(month=m, year='2014', tile="h25v07" outputfolder="/media/kiruba/New Volume/MODIS/ET/scratch")

# multiply to get et, exclude other values
# reproject
# save as geotiff


