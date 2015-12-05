__author__ = 'kiruba'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import os
import sys



gisbase = os.environ['GISBASE'] = "/usr/lib/grass70"
gisdbase = os.path.join(os.environ['HOME'], "grass")
location = "aralu_utm"
mapset = "aralu_delin"
sys.path.append(os.path.join(os.environ['GISBASE'], "etc", "python"))

import grass.script as grass
import grass.script.setup as gsetup


# setting up grass with set parameters
gsetup.init(gisbase, gisdbase, location, mapset)
print grass.gisenv()
# DEM for tg_halli
dem = "/media/kiruba/New Volume/milli_watershed/check_dam_catchment_area/20150723044746_438840891_utm.tif"
dem_title = "tg_halli_dem"
# Input raster into grass
# grass.run_command("r.in.gdal", input=dem, output=dem_title )
grass.run_command("g.region", rast=dem_title, flags='p')

aralu_utm_vector = "/media/kiruba/New Volume/milli_watershed/aralumallige/milli_aralumallige.shp"
aralu_utm_vector_title = "aralu_utm"
# Input vector into grass
# grass.run_command("v.in.ogr", dsn=aralu_utm_vector, output=aralu_utm_vector_title )
# check dam vector
aralu_check_dam_file = "/media/kiruba/New Volume/milli_watershed/check_dam/ch_aral_utm.shp"
aralu_check_dam = "aralu_check_dam_utm"
# remove vector
grass.run_command('g.remove', vect=aralu_check_dam, quiet=True)
# Input vector into grass
grass.run_command("v.in.ogr", dsn=aralu_check_dam_file, output=aralu_check_dam, type='point' )
# set region settings to match vector
grass.run_command('g.region', vect=aralu_utm_vector_title, flags='p')
# list all the vectors within mapset
grass.message("Vector maps:")
for vect in grass.list_strings(type="vect"):
    print vect
"""
Clip raster with vector
"""
aralu_utm_raster = "aralu_utm_rast"
# remove raster
# grass.run_command('g.remove', rast=aralu_utm_raster, quiet=True)
# rasterize vector to raster for raster-raster clipping using r.mapcalc
# grass.run_command("v.to.rast", input=aralu_utm_vector_title, output=aralu_utm_raster,type='area', use='val', value=1 )
# grass.run_command('r.info', map=aralu_utm_raster)
# raise SystemExit(0)

aralu_dem_utm = "aralu_dem_utm"
# remove raster
grass.run_command('g.remove', rast=aralu_dem_utm, quiet=True)
# clip raster
grass.mapcalc('%s = if(%s, %s)' % (aralu_dem_utm, aralu_utm_raster, dem_title))
# alternate method
# grass.run_command('r.mask', flags='r')
# grass.run_command('r.mask', input=aralu_utm_raster)
# grass.run_command('g.copy', rast=(dem_title, aralu_dem_utm))
"""
Watershed map
"""
# aster dem resolution 30 m, so pixel size is 900m2 or 0.0009 km2. to set flow acc area as 2 sq km set threshold 2222 pixels
# 2/0.0009 = 2222.2222
# 2222 threshold is giving very coarse, so lets try 1 sqkm 1111
# 1111 threshold is giving very coarse, so lets try 0.5 sqkm 555
# 555 threshold is giving very coarse, so lets try 0.25 sqkm 227
# create drainage map using r.watershed which will be used as input r,water,outlet
accumulation = "accum_227"
drainage = "drainage_227"
basin = "catch_227"
stream = "stream_227"
# remove raster
grass.run_command('g.remove', rast=accumulation, quiet=True)
grass.run_command('g.remove', rast=drainage, quiet=True)
grass.run_command('g.remove', rast=basin, quiet=True)
grass.run_command('g.remove', rast=stream, quiet=True)
grass.run_command("r.watershed",elevation=aralu_dem_utm, threshold=227, accumulation=accumulation, drainage=drainage,
                  basin=basin, stream=stream)
# Drain points for check dam in aralumallige
cols = ['easting', 'northing']
check_dam_no = [599, 591, 618, 612]
#  599 772437.354,1466201.025 | 591 772389.922,1467544.293 | 618 774518.660,1466204.820  |  616 775338.282,1465695.402
# 612 773467.572,1467608.800 |
# easting_series = ##599##    ###591###   ###618###   ###612###
easting_series = [772437.354, 772389.922, 774518.660, 773467.572]
northing_series = [1466201.025, 1467544.293, 1466204.820, 1467608.800]
series = [pd.Series(easting_series,index=check_dam_no,name=cols[0]), pd.Series(northing_series, index=check_dam_no, name=cols[1])]
df = pd.concat(series, axis=1, keys=cols)
# r.water.outlet
# remove previous rasters
# for c in check_dam_no:
#     grass.run_command('g.remove', rast='test1_%s' % c, quiet=True)
# the input for r.water.outlet is drainage direction map1467185.874,
# for c in check_dam_no:
#     grass.run_command("r.water.outlet", drainage=drainage, basin='test1_%s' %c, easting=df.loc[c, 'easting'], northing=df.loc[c,'northing'])
# # display raster
# grass.run_command('d.mon', flags="l", start='x2')
# grass.run_command('d.rast',map=aralu_dem_utm)
# display vector

# grass.run_command("g.region", rast=aralu_dem_utm, flags='p')
# grass.run_command('r.info', map=dem_title)
# grass.run_command('v.info', map=aralu_check_dam)
# grass.run_command('d.mon', flags="l", start='x0')
# # grass.run_command('d.rast',map=aralu_dem_utm)
# grass.run_command('d.rast',map=drainage)
# grass.run_command('d.rast',map=stream, flags='o')
# # display vector
# grass.run_command('d.vect', map=aralu_utm_vector_title, type='boundary', width=3, color='red')
# grass.run_command('d.vect', map=aralu_check_dam, type='point', color='black',fcolor='white',  icon='basic/point', size=30, width=3)
# grass.run_command('d.zoom')

# for c in check_dam_no:
#     print c
#     grass.run_command('d.rast', flags='o', map='test1_%s' % c)
# convert raster to vector
# for c in check_dam_no:
#     grass.run_command('r.to.vect', input='test1_%s' %c, output='catchment_%s' %c, feature='area')

# for c in check_dam_no:
#     grass.run_command('d.vect', map='catchment_%s' % c, type='boundary', width=3, color='green')

# for c in check_dam_no:
#     print c
#     grass.run_command('v.to.db', map='catchment_%s' %c, type='boundary', option='area', units='k', flags='p')
"""
hadonahalli
"""
grass.run_command("g.region", rast=dem_title, flags='p')

had_utm_vector_file = "/media/kiruba/New Volume/milli_watershed/hadonahalli/hadonahalli_new_utm.shp"
had_utm_vector = "had_utm"
# Input vector into grass
grass.run_command("v.in.ogr", dsn=had_utm_vector_file, output=had_utm_vector, type='boundary')
# check dam vector
had_check_dam_file = "/media/kiruba/New Volume/milli_watershed/check_dam/ch_had_utm.shp"
had_check_dam = "had_check_dam_utm"
# remove vector
# grass.run_command('g.remove', vect=had_check_dam, quiet=True)
# Input vector into grass
grass.run_command("v.in.ogr", dsn=had_check_dam_file, output=had_check_dam, type='point' )
# set region settings to match vector
grass.run_command('g.region', vect=had_utm_vector, flags='p')
# list all the vectors within mapset
grass.message("Vector maps:")
for vect in grass.list_strings(type="vect"):
    print vect
"""
Clip raster with vector
"""
had_utm_raster = "had_utm_rast"
# remove raster
# grass.run_command('g.remove', rast=aralu_utm_raster, quiet=True)
# rasterize vector to raster for raster-raster clipping using r.mapcalc
grass.run_command("v.to.rast", input=had_utm_vector, output=had_utm_raster,type='area', use='val', value=1 )
# grass.run_command('r.info', map=aralu_utm_raster)
# raise SystemExit(0)
had_dem_utm = "had_dem_utm"
# remove raster
# grass.run_command('g.remove', rast=aralu_dem_utm, quiet=True)
# clip raster
grass.mapcalc('%s = if(%s, %s)' % (had_dem_utm, had_utm_raster, dem_title))

"""
Watershed map
"""
# create drainage map using r.watershed which will be used as input r,water,outlet
accumulation = "had_accum_115"
drainage = "had_drainage_115"
basin = "had_catch_115"
stream = "had_stream_115"
# remove raster
# grass.run_command('g.remove', rast=accumulation, quiet=True)
# grass.run_command('g.remove', rast=drainage, quiet=True)
# grass.run_command('g.remove', rast=basin, quiet=True)
# grass.run_command('g.remove', rast=stream, quiet=True)
# grass.run_command("r.watershed",elevation=had_dem_utm, threshold=115, accumulation=accumulation, drainage=drainage,
#                   basin=basin, stream=stream)

# Drain points for check dam in hadonahalli
cols = ['easting', 'northing']
check_dam_no = [640, 625, 641, 634]
#  640 773890.660,1478378.864 | 625 776781.355,1478865.637 | 641 774983.703,1477732.216 | 634 777061.916,1479054.119
# easting_series = ##640##    ###625###   ###641###  ###634###
easting_series = [773890.660, 776781.355, 774983.703, 777061.916]
northing_series = [1478378.864, 1478865.637, 1477732.216, 1479054.119]
series = [pd.Series(easting_series,index=check_dam_no,name=cols[0]), pd.Series(northing_series, index=check_dam_no, name=cols[1])]
df = pd.concat(series, axis=1, keys=cols)
# r.water.outlet
# remove previous rasters
# for c in check_dam_no:
#     grass.run_command('g.remove', rast='test1_%s' % c, quiet=True)
# the input for r.water.outlet is drainage direction map
# for c in check_dam_no:
#     grass.run_command("r.water.outlet", drainage=drainage, basin='catchment_%s' %c, easting=df.loc[c, 'easting'], northing=df.loc[c,'northing'])

for c in check_dam_no:
    grass.run_command('r.to.vect', input='catchment_%s' %c, output='catchment_%s' %c, feature='area')

for c in check_dam_no:
    print c
    grass.run_command('v.to.db', map='catchment_%s' %c, type='boundary', option='area', units='k', flags='p')

# list all the rasters within mapset
grass.message("Raster maps:")
for rast in grass.list_strings(type="rast"):
    print rast


