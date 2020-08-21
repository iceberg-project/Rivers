"""
Authors: Samira Daneshgar-Asl
License: MIT
Copyright: 2019-2020
"""
import os
import numpy as np
from osgeo import gdal
		
#loading image 
def load_image(path):
    ds = gdal.Open(path) 
    img_proj = ds.GetProjection() 
    img_geotrans = ds.GetGeoTransform()
    img_data = ds.ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize)  
    del ds
    image = np.array(img_data,dtype=img_data.dtype)
    return img_proj, img_geotrans, image

#writing mosaic 
def write_mosaic(filename,img_proj,img_geotrans,img_data):    
    driver = gdal.GetDriverByName("GTiff")
    bands, (ysize, xsize) = 1, img_data.shape  
    ds = driver.Create(filename, xsize, ysize, bands, gdal.GDT_Float32)
    ds.SetProjection(img_proj)
    ds.SetGeoTransform(img_geotrans)                          
    ds.GetRasterBand(1).WriteArray(img_data)

def mosaic_unet(input_path, output_path, step, input_WV, tile_size):
    masks_path = input_path
    pred_list = sorted(os.listdir(masks_path),key=lambda x: int(os.path.splitext(x)[0]))
    image_name = input_WV.split('/')[-1].split('.')[0]
    out_path = output_path 
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    proj, geotrans, image = load_image(input_WV)
    desired_row_size = np.int(step * (np.ceil(image.shape[1] / step) + 1))
    desired_col_size = np.int(step * (np.ceil(image.shape[2] / step) + 1))
    mask = np.zeros((desired_row_size, desired_col_size), dtype=np.float64)

    k=0
    for j in range(0, mask.shape[1]-(tile_size-step), step):
        for i in range(0, mask.shape[0] - (tile_size - step), step):
            mask_name = pred_list[k]
            mask_proj, mask_geotranse, mask_tile= load_image(masks_path + mask_name) 			
            mask[i:i + tile_size, j:j +     tile_size] = np.maximum(mask_tile[:, :], mask[i: i + tile_size, j:j + tile_size])
            k+=1
    write_mosaic(out_path + "%s_predicted.tif" % image_name, proj, geotrans,
                 mask[0:image.shape[1], 0:image.shape[2]])   
