import arcpy,os,sys
from arcpy import env
from arcpy.sa import *

#set the workspace and list all of the raster dataset
#env.workspace=r'D:\Analysis\Greenland_Analysis\GreenlandHighRes\StreamExtraction\multioutput\watermask\toprocess'
env.workspace=r'F:\Courtney\Greenland_Code\Courtney_Stream_Extraction\multioutput\watermask'
#r'D:\2012Images\WorldView\geotiff\730\resample\watermask'
env.overwriteOutput=True
#output=r'D:\Analysis\Greenland_Analysis\GreenlandHighRes\StreamExtraction\multioutput\watermask\toprocess\output'
img_output=r'F:\Courtney\Greenland_Code\Courtney_Stream_Extraction\multioutput\thinned_img'
shp_output=r'F:\Courtney\Greenland_Code\Courtney_Stream_Extraction\multioutput\thinned_shapefile'
#r'D:\2012Images\WorldView\geotiff\730\resample\watermask\output'

arcpy.CheckOutExtension("Spatial")

tiffs = arcpy.ListRasters("*","img")
print tiffs
arcpy.CheckOutExtension("Spatial")
for tiff in tiffs:
    print "start process " + tiff
    outThinnedRaster=img_output+ "\\"+(tiff.split('.'))[0]+"thin.img"
    outFeatureClass=shp_output+ "\\"+(tiff.split('.'))[0]+".shp"
    thinOut=Thin(tiff, "ZERO","NO_FILTER","ROUND")
    thinOut.save(outThinnedRaster)
    arcpy.RasterToPolyline_conversion(outThinnedRaster,outFeatureClass,"ZERO",50,"NO_SIMPLIFY","")
