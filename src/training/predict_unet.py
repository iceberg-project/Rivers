"""
Authors: Samira Daneshgar-Asl
License: MIT
Copyright: 2019-2020
"""
import os
import numpy as np
import argparse
from model import *
from keras.preprocessing.image import img_to_array
from osgeo import gdal
from os import listdir
		
def adjustTile(tile):
    if(np.max(tile) > 1):
        tile = tile / 255
    return (tile)
	
#loading tiles
def load_tile(path):
    ds = gdal.Open(path)
	xsize = ds.RasterXSize    
    ysize = ds.RasterYSize
    band3=ds.GetRasterBand(3)
    band5=ds.GetRasterBand(5)
    band6=ds.GetRasterBand(6)       
    tile_proj = ds.GetProjection()
    tile_geotrans = ds.GetGeoTransform()
    b3,b5,b6 = (band3.ReadAsArray(0,0, xsize, ysize), band5.ReadAsArray(0,0, xsize, ysize), band6.ReadAsArray(0,0, xsize, ysize)) 
    del ds 
    tile = np.array(np.stack([b3, b5, b6]), dtype=b3.dtype)
    tile = np.swapaxes(tile,0,1)
    tile = np.swapaxes(tile,1,2)
    tile = adjustTile(tile) 
    return tile_proj, tile_geotrans, tile
	
#writing predicted masks 
def write_mask(filename,mask_proj,mask_geotrans,mask_data):    
    driver = gdal.GetDriverByName("GTiff")
    bands, (ysize, xsize) = 1,mask_data.shape   
    ds = driver.Create(filename, xsize, ysize, bands, gdal.GDT_Float32)
    ds.SetProjection(mask_proj)
    ds.SetGeoTransform(mask_geotrans)                          
    ds.GetRasterBand(1).WriteArray(mask_data)  
    del ds
    
def args_parser():
    parser = argparse.ArgumentParser(description="predicts tiles")
    parser.add_argument('-i', '--input', type=str,
                            help='Input image tiles folder')
    parser.add_argument('-w', '--weights_path', type=str,
                            help='Path to the weights')
    parser.add_argument('-o', '--output_folder', type=str, default='./',
                        help='Folder where output masks will be stored')			
    return parser.parse_args()
    
if __name__ == '__main__':
    args = args_parser()

    model = unet()
    model.load_weights(args.weights_path)
    tiles_path = 'image_tiles/' + args.input + '/' 
    list = os.listdir(tiles_path) 
    num_tiles = len(list)
	
    out_path = 'predicted_tiles/' + args.input + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    for n in range(num_tiles):
        tile_name = list[n]
        head, tail = tile_name.split('.tif')
        proj, geotrans, tile= load_tile(tiles_path + tile_name)
        array_tile = img_to_array(tile)
        expand_tile = np.expand_dims(array_tile, axis=0)       	
        pred = model.predict(expand_tile)
        for i,item in enumerate(pred):
            mask = item[:,:,0]
        write_mask(out_path+ "%s_predicted.tif"%head, proj, geotrans, mask)
	
	
	
