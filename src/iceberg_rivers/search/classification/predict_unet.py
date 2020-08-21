"""
Authors: Samira Daneshgar-Asl
License: MIT
Copyright: 2019-2020
"""
import os
import numpy as np
from .model import *
from keras.preprocessing.image import img_to_array
from osgeo import gdal
		
def adjustTile(img):
    if(np.max(img) > 1):
        img = img / 255
    return (img)
	
#loading tiles
def load_tile(path):
    ds = gdal.Open(path)
    band3 = ds.GetRasterBand(3)
    band5 = ds.GetRasterBand(5)
    band6 = ds.GetRasterBand(6)       
    img_proj = ds.GetProjection()
    img_geotrans = ds.GetGeoTransform()
    b3,b5,b6 = (band3.ReadAsArray(0,0, ds.RasterXSize, ds.RasterYSize),
                band5.ReadAsArray(0,0, ds.RasterXSize, ds.RasterYSize),
                band6.ReadAsArray(0,0, ds.RasterXSize, ds.RasterYSize)) 
    del ds 
    image = np.array(np.stack([b3, b5, b6]), dtype=b3.dtype)
    image = np.swapaxes(image,0,1)
    image = np.swapaxes(image,1,2)
    image = adjustTile(image) 
    return img_proj, img_geotrans, image
	
#writing predicted masks 
def write_mask(filename,img_proj,img_geotrans,img_data):    
    driver = gdal.GetDriverByName("GTiff")
    bands, (ysize, xsize) = 1,img_data.shape  
    ds = driver.Create(filename, xsize, ysize, bands, gdal.GDT_Float32)
    ds.SetProjection(img_proj)
    ds.SetGeoTransform(img_geotrans)                          
    ds.GetRasterBand(1).WriteArray(img_data)
    
def predict_unet(input, weights_path, output_folder):

    model = unet()
    model.load_weights(weights_path)
    tiles_path = input
    list = os.listdir(tiles_path) 
    num_tiles = len(list)

    print(tiles_path, num_tiles)	
    out_path = output_folder
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
        write_mask(out_path + "%s.tif" % head, proj, geotrans, mask)
