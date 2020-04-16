"""
Authors: Samira Daneshgar-Asl
License: MIT
Copyright: 2019-2020
"""
import os, gdal
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                            help='Path and Filename of the WV Image')
    parser.add_argument('-o', '--output', type=str,
                            help='Output name')
    args = parser.parse_args()
 
    out_path = 'image_tiles/' + args.output + "/"
    if not os.path.exists(out_path):
       os.makedirs(out_path)
 
    tile_size_x = 256
    tile_size_y = 256
    step=128
 
    ds = gdal.Open(args.input)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize

    k=0 
    for i in range(0, xsize, step):
        for j in range(0, ysize, step):
            com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(args.input) + " " + str(out_path) + str(k) + ".tif"
            os.system(com_string)
            k+=1


if __name__ == '__main__':
    main()
