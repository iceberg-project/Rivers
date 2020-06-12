"""
Authors: Samira Daneshgar-Asl
License: MIT
Copyright: 2019-2020
"""
import os
import numpy as np
import argparse
import math
from osgeo import gdal
from os import listdir
		
#loading image 
def load_image(path):
    ds = gdal.Open(path) 
    img_proj = ds.GetProjection() 
    img_geotrans = ds.GetGeoTransform()
    img_data = ds.ReadAsArray(0,0, ds.RasterXSize, ds.RasterYSize)  
    del ds
    image = np.array(img_data,dtype=img_data.dtype)
    return img_proj,img_geotrans,image

#writing mosaic 
def write_mosaic(filename,img_proj,img_geotrans,img_data):    
    driver = gdal.GetDriverByName("GTiff")
    bands, (ysize, xsize) = 1,img_data.shape  
    ds = driver.Create(filename, xsize, ysize, bands, gdal.GDT_Float32)
    ds.SetProjection(img_proj)
    ds.SetGeoTransform(img_geotrans)                          
    ds.GetRasterBand(1).WriteArray(img_data)  
    
def args_parser():
    parser = argparse.ArgumentParser(description="generates mosaic")
    parser.add_argument('-iw', '--input_WV', type=str,
                        help='Path and name of the WorldView image')
    parser.add_argument('-i', '--input', type=str,
                            help='Input predicted masks folder')
    parser.add_argument('-t', '--tile_size', type=int, default=224,
                        help='Tile size')
    parser.add_argument('-s', '--step', type=int, default=112,
                        help='Step size')						 
    parser.add_argument('-o', '--output_folder', type=str, default='./',
                        help='Folder where output mosaic will be stored')			
    return parser.parse_args()
    
if __name__ == '__main__':
    args = args_parser()
	
    masks_path = 'predicted_tiles/' + args.input + '/' 
    list = sorted(os.listdir(masks_path),key=lambda x: int(os.path.splitext(x)[0]))

    out_path = 'predicted_mosaic/' + args.output_folder 
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    proj, geotrans, image = load_image(args.input_WV)
    desired_row_size = args.step * (math.ceil(image.shape[1]/ args.step)+1)
    desired_col_size = args.step * (math.ceil(image.shape[2]/ args.step)+1)
    mask = np.zeros((desired_row_size,desired_col_size),dtype=np.float64)

    k=0
    for j in range(0, mask.shape[1]-(args.tile_size-args.step), args.step):
        for i in range(0, mask.shape[0]-(args.tile_size-args.step), args.step):
            mask_name = list[k]
            mask_proj, mask_geotranse, mask_tile= load_image(masks_path + mask_name) 			
            mask[i:i + args.tile_size, j:j + args.tile_size]=np.maximum(mask_tile[:,:],mask[i:i + args.tile_size, j:j + args.tile_size])
            k+=1
    write_mosaic(out_path+ "%s_predicted.tif"%args.input, proj, geotrans, mask[0:image.shape[1], 0:image.shape[2]])


