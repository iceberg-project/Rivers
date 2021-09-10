"""
Authors: Samira Daneshgar-Asl
License: MIT
Copyright: 2020-2021
"""
import os
import numpy as np
import argparse
from model import *
from keras.preprocessing.image import img_to_array
from osgeo import gdal
from os import listdir
import time

np.seterr(divide='ignore', invalid='ignore')

def normalize_multi(img):
    if(np.max(img) > 1):
        img = img / 255
    return (img)


def normalize_pan(img):
    img = img / 65535
    average = img[np.nonzero(img)].mean()
    standard = img[np.nonzero(img)].std()
    x = (img - average)/ standard
    return x


#loading tiles
def load_tile(path):
    ds = gdal.Open(path)
    img_proj = ds.GetProjection()
    img_geotrans = ds.GetGeoTransform()
    img_data = ds.ReadAsArray(0,0, ds.RasterXSize, ds.RasterYSize)
    del ds
    image = np.array(img_data,dtype=img_data.dtype)
    if len(image.shape)==2: #if grayscale image
        image = normalize_pan(image)
    else: #if not grayscale image, change the order of the axis
        image = np.swapaxes(image,0,1)
        image = np.swapaxes(image,1,2)
        image = normalize_multi(image)
    return img_proj, img_geotrans, image


#writing predicted tiles
def write_mask(filename,img_proj,img_geotrans,img_data):
    driver = gdal.GetDriverByName("GTiff")
    bands, (ysize, xsize) = 1,img_data.shape
    ds = driver.Create(filename, xsize, ysize, bands, gdal.GDT_Float32)
    ds.SetProjection(img_proj)
    ds.SetGeoTransform(img_geotrans)
    ds.GetRasterBand(1).WriteArray(img_data)


def args_parser():
    parser = argparse.ArgumentParser(description="predicts tiles")
    parser.add_argument('--input_folder', '-i', type=str, required=True,
                        help='folder where tiles are stored')
    parser.add_argument('--weights_path', '-w', type=str, required=True,
                        help='full path to weights')
    parser.add_argument('--output_folder', '-o', type=str, required=True,
                        help='folder where predicted tiles will be stored')
    return parser.parse_args()

if __name__ == '__main__':
    args = args_parser()

    model = unet()
    model.load_weights(args.weights_path)

    # time it
    tic = time.time()

    tiles_path = '%s/' % (args.input_folder)
    list = os.listdir(tiles_path)
    num_tiles = len(list)

    out_path = '%s/' % (args.output_folder)
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
        write_mask(out_path+ "%s.tif"%head, proj, geotrans, mask)

    elapsed = time.time() - tic
    print('predicted tiles created in %d minutes and %.2f seconds' % (int(elapsed // 60), elapsed % 60))
