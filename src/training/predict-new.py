"""
Authors: Samira Daneshgar-Asl
License: MIT
Copyright: 2019-2020
"""
import os
import numpy as np
import argparse
from model import *
from os import listdir
from keras.preprocessing.image import img_to_array
from osgeo import gdal

Nonriver = [255,255,255]
River = [0,0,0]
COLOR_DICT = np.array([Nonriver,River])
		
patch_sz = 256

#loading tiles 
def load_tile(path):
    dataset = gdal.Open(path)      
    tile_geotrans = dataset.GetGeoTransform()  
    tile_proj = dataset.GetProjection()
    tile_data = dataset.ReadAsArray(0,0,patch_sz,patch_sz) 
    del dataset 
    tile = np.array(tile_data,dtype=tile_data.dtype) 
    tile = np.swapaxes(tile,0,1)
    tile = np.swapaxes(tile,1,2)
    tile =tile/255 
    return tile_geotrans,tile_proj, tile
	
#writing predicted masks 
def write_tile(filename,tile_proj,tile_geotrans,tile_data):    
    mask_bands = 1 
    driver = gdal.GetDriverByName("GTiff")   
    dataset = driver.Create(filename, patch_sz, patch_sz, mask_bands, gdal.GDT_Byte)
    dataset.SetGeoTransform(tile_geotrans)          
    dataset.SetProjection(tile_proj)                
    dataset.GetRasterBand(1).WriteArray(tile_data)  
    del dataset
    
def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                            help='Input image tiles folder')
    parser.add_argument('-w', '--weights_path', type=str,
                            help='Path to the weights')
    
    args = parser.parse_args() 
    model = unet()
    model.load_weights(args.weights_path) 
    tiles_path = 'image_tiles/' + args.input + '/'
    list = os.listdir(tiles_path) 
    num_tiles = len(list)
    
    outPath = 'predicted_tiles/' + args.input + '/' 
    if not os.path.exists(outPath):
       os.makedirs(outPath)
    
    for n in range(num_tiles):
        tile_name = list[n]
        head, tail = tile_name.split('.tif')
        #print(tile_name)
        geotrans, proj, tile = load_tile(tiles_path + tile_name)
        array_tile = img_to_array(tile)
        expand_tile = np.expand_dims(array_tile, axis=0)       	
        pred = model.predict(expand_tile)
        for i,item in enumerate(pred):
            mask = item[:,:,0]
            mask[mask>0.3]=255
            mask[mask<=0.3]=0
        #mask=pred[0,:,:,0]
        write_tile(outPath+ "%s_predicted.tif"%head, proj, geotrans, mask)
		
if __name__ == '__main__':
    predict()
    
       
