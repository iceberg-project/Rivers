"""
Authors: Samira Daneshgar-Asl
License: MIT
Copyright: 2019-2020
"""
import os
from osgeo import gdal


def tile_unet(image_path, ouput_path, step):

    
    if not os.path.exists(out_path):
       os.makedirs(out_path)
 
    image = gdal.Open(image_path)	

    k=0 
    for i in range(0, image.RasterXSize, step):
        for j in range(0, image.RasterYSize, step):
            com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(args.tile_size) + ", " + str(args.tile_size) + " " + str(args.input) + " " + str(out_path) + str(k) + ".tif"
            os.system(com_string)
            k+=1
