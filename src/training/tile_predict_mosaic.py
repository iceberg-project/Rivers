"""
Authors: Samira Daneshgar-Asl
License: MIT
Copyright: 2019-2020
"""
import os
import numpy as np
import argparse
import math
from model import *
from os import listdir
from keras.preprocessing.image import img_to_array
from osgeo import gdal
from osgeo import gdal_array
		
patch_sz = 256
step_sz = 128

#loading WV image 
def load_image(path):
    dataset = gdal.Open(path) 
    image_column = dataset.RasterXSize    
    image_row = dataset.RasterYSize     
    image_proj = dataset.GetProjection()
    image_geotrans = dataset.GetGeoTransform()
    image_data = dataset.ReadAsArray(0,0,image_column,image_row) 
    del dataset 
    image = np.array(image_data,dtype=image_data.dtype)
    image = np.swapaxes(image,0,1)
    image = np.swapaxes(image,1,2)
    image =image/255 
    return image_geotrans,image_proj, image
	
#writing predicted mask 
def write_mask(filename,mask_proj,mask_geotrans,mask_data):    
    driver = gdal.GetDriverByName("GTiff")
    mask_bands, (mask_row, mask_column) = 1,mask_data.shape   
    dataset = driver.Create(filename, mask_column, mask_row, mask_bands, gdal.GDT_Float32)
    dataset.SetProjection(mask_proj)
    dataset.SetGeoTransform(mask_geotrans)                          
    dataset.GetRasterBand(1).WriteArray(mask_data)  
    del dataset
    
def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                            help='Input image folder')
    parser.add_argument('-w', '--weights_path', type=str,
                            help='Path to the weights')
    parser.add_argument('-o', '--output_folder', type=str, default='./',
                        help='Path where output will be stored')

    args = parser.parse_args() 
    model = unet()
    model.load_weights(args.weights_path)
    head, tail = os.path.split(args.input)
    getName = tail.split('-3bands-356.tif')
    outPath = args.output_folder + "predicted_image/" 
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    
    geotrans, proj, image = load_image(args.input)

    desired_row_size = step_sz * math.ceil(image.shape[0]/ step_sz)
    desired_col_size = step_sz * math.ceil(image.shape[1]/ step_sz)
    desired_image = np.zeros((desired_row_size, desired_col_size, image.shape[2]),
                   dtype=image.dtype)
    desired_image[:image.shape[0], :image.shape[1],:] = image[:,:,:]
    desired_image = img_to_array(desired_image)
    
    mask = np.zeros((desired_row_size,desired_col_size),dtype=np.float64)

    for i in range(0, image.shape[0]-step_sz, step_sz):
        for j in range(0, image.shape[1]-step_sz, step_sz):
            # the image which has to be predicted
            tile = desired_image[i:i + patch_sz, j:j + patch_sz,:]
            tile = np.expand_dims(tile, axis=0)
            pred = model.predict(tile)
            for m,item in enumerate(pred):
                tile_mask = 1-item[:,:,0]
                #tile_mask[tile_mask>=0.85]=255
                #tile_mask[tile_mask<0.85]=0
            #tile_mask=pred[0,:,:,0]    
            mask[i:i + patch_sz, j:j + patch_sz]=np.maximum(tile_mask[:,:],mask[i:i + patch_sz, j:j + patch_sz])
    write_mask(outPath+ "%s_predicted.tif"%getName[0], proj, geotrans, mask[0:image.shape[0], 0:image.shape[1]])

if __name__ == '__main__':                     
    predict()  
       
