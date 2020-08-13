"""
Authors: Samira Daneshgar-Asl
License: MIT
Copyright: 2019-2020
"""
import os
from osgeo import gdal


def tile_unet(image_path, output_path, step, tile_size):

    
    if not os.path.exists(output_path):
       os.makedirs(output_path)
 
    image = gdal.Open(image_path)	

    k=0 
    for i in range(0, image.RasterXSize, step):
        for j in range(0, image.RasterYSize, step):
            com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size) + ", " + str(tile_size) + " " + str(image_path) + " " + str(output_path) + str(k) + ".tif"
            os.system(com_string)
            k+=1
