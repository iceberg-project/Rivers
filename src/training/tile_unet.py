"""
Authors: Samira Daneshgar-Asl
License: MIT
Copyright: 2019-2020
"""
import os
import argparse
from osgeo import gdal

def args_parser():
    parser = argparse.ArgumentParser(description="divides a WorldView image into tiles")
    parser.add_argument('-i', '--input', type=str,
                        help='Path and name of the WorldView image')
    parser.add_argument('-t', '--tile_size', type=int, default=224,
                        help='Tile size')
    parser.add_argument('-s', '--step', type=int, default=112,
                        help='Step size')
    parser.add_argument('-o', '--output', type=str, 
                        help='Folder where output tiles will be stored')			
    return parser.parse_args()
    
if __name__ == '__main__':
    args = args_parser()
 
    out_path = 'image_tiles/' + args.output + "/"
    if not os.path.exists(out_path):
       os.makedirs(out_path)
 
    image = gdal.Open(args.input)	

    k=0 
    for i in range(0, image.RasterXSize, args.step):
        for j in range(0, image.RasterYSize, args.step):
            com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(args.tile_size) + ", " + str(args.tile_size) + " " + str(args.input) + " " + str(out_path) + str(k) + ".tif"
            os.system(com_string)
            k+=1

