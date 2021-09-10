"""
Authors: Samira Daneshgar-Asl
License: MIT
Copyright: 2020-2021
"""
import os
import numpy as np
import argparse
import math
from osgeo import gdal
from os import listdir
import time

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
    parser.add_argument('--input_image', '-iw', type=str, required=True,
                        help='full path to raster file we tiled out')
    parser.add_argument('--input_folder', '-i', type=str, required=True,
                        help='folder where predicted tiles are stored')
    parser.add_argument('--patch_size', '-p', type=int, default=256, required=False,
                        help='side dimensions for each patch. patches are required to be squares.')
    parser.add_argument('--stride', '-s', type=float, default=0.5, required=False,
                        help='distance between tiles as a multiple of patch_size. defaults to 0.5 for 50% overlap'
                             'tiles without overlap')
    parser.add_argument('--output_folder', '-o', type=str, required=True,
                        help='folder where predicted raster file will be stored')

    return parser.parse_args()

if __name__ == '__main__':
    args = args_parser()

    # time it
    tic = time.time()

    masks_path = '%s/' % (args.input_folder)
    list = sorted(os.listdir(masks_path),key=lambda x: int(os.path.splitext(x)[0]))

    out_path = '%s/' % (args.output_folder)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    head, tail = os.path.split(args.input_image)
    getName = tail.split('.tif')

    proj, geotrans, image = load_image(args.input_image)

    if len(image.shape)==2:
        desired_row_size = int(args.stride*args.patch_size) * (math.ceil(image.shape[0]/ int(args.stride*args.patch_size))+1)
        desired_col_size = int(args.stride*args.patch_size) * (math.ceil(image.shape[1]/ int(args.stride*args.patch_size))+1)
    else:
        desired_row_size = int(args.stride*args.patch_size) * (math.ceil(image.shape[1]/ int(args.stride*args.patch_size))+1)
        desired_col_size = int(args.stride*args.patch_size) * (math.ceil(image.shape[2]/ int(args.stride*args.patch_size))+1)
    mask = np.zeros((desired_row_size,desired_col_size),dtype=np.float32)

    k=0
    for i in range(0, mask.shape[0]-(args.patch_size-int(args.stride*args.patch_size)), int(args.stride*args.patch_size)):
        for j in range(0, mask.shape[1]-(args.patch_size-int(args.stride*args.patch_size)), int(args.stride*args.patch_size)):
            mask_name = list[k]
            mask_proj, mask_geotranse, mask_tile= load_image(masks_path + mask_name)
            mask[i:i + args.patch_size, j:j + args.patch_size]=np.maximum(mask_tile[:,:],mask[i:i + args.patch_size, j:j + args.patch_size])
            k+=1
        if len(image.shape)==2:
            write_mosaic(out_path+ "%s_predicted.tif"%getName[0], proj, geotrans, mask[0:image.shape[0], 0:image.shape[1]])
        else:
            write_mosaic(out_path+ "%s_predicted.tif"%getName[0], proj, geotrans, mask[0:image.shape[1], 0:image.shape[2]])

    elapsed = time.time() - tic
    print('predicted raster file created in %d minutes and %.2f seconds' % (int(elapsed // 60), elapsed % 60))

