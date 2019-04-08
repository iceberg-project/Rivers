"""
Author: Samira Daneshgar-Asl
License: MIT
Copyright: 2018-2019
"""

import math
import numpy as np
import tifffile as tiff
import os

from train_unet import weights_path, get_model, normalize, PATCH_SZ, N_CLASSES

if not os.path.exists('data/WV_predicted'):
        os.makedirs('data/WV_predicted')

image= normalize(tiff.imread('name of the WV image.tif').transpose([1,2,0]))
wind_row, wind_col = 800,800 # dimensions of the image
windowSize = 800 
stepSize=400


desired_size_row=stepSize*math.ceil(image.shape[0]/stepSize)
desired_size_col=stepSize*math.ceil(image.shape[1]/stepSize)

img = np.zeros((desired_size_row,desired_size_col,image.shape[2]), dtype=image.dtype)

img[:image.shape[0],:image.shape[1]] = image


def predict(x, model, patch_sz=160, n_classes=2):
    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]
    # make extended img so that it contains integer number of patches
    npatches_vertical = math.ceil(img_height / patch_sz)
    npatches_horizontal = math.ceil(img_width / patch_sz)
    extended_height = patch_sz * npatches_vertical
    extended_width = patch_sz * npatches_horizontal
    ext_x = np.zeros(shape=(extended_height, extended_width, n_channels), dtype=np.float32)
    # fill extended image with mirrors:
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2 * img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2 * img_width - j - 1, :]

    # now we assemble all patches in one array
    patches_list = []
    for i in range(0, npatches_vertical):
        for j in range(0, npatches_horizontal):
            x0, x1 = i * patch_sz, (i + 1) * patch_sz
            y0, y1 = j * patch_sz, (j + 1) * patch_sz
            patches_list.append(ext_x[x0:x1, y0:y1, :])
    # model.predict() needs numpy array rather than a list
    patches_array = np.asarray(patches_list)
    # predictions:
    patches_predict = model.predict(patches_array, batch_size=4)
    prediction = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float32)
    for k in range(patches_predict.shape[0]):
        i = k // npatches_horizontal
        j = k % npatches_vertical
        x0, x1 = i * patch_sz, (i + 1) * patch_sz
        y0, y1 = j * patch_sz, (j + 1) * patch_sz
        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
    return prediction[:img_height, :img_width, :]

# generating sliding window
def sliding_window(img, stepSize, windowSize):
    for y in range(0, img.shape[0], stepSize):
        for x in range(0, img.shape[1], stepSize):
            yield (x, y, img[y:y + windowSize, x:x + windowSize,:])
                        
def main():
    outPath = 'data/WV_predicted'
    i=0
    for(x,y, window) in sliding_window(img, stepSize, windowSize):
        if window.shape[0] != wind_row or window.shape[1] != wind_col:
            continue
        t_img = img[y:y+wind_row,x:x+wind_col,:]# the image which has to be predicted
        mask = predict(t_img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1]) 
        cnt=str(i)
        imagename="image-"+cnt+".tif"
        fullpath = os.path.join(outPath,imagename)
        tiff.imsave(fullpath, mask)
        i=i+1
        
if __name__ == '__main__':
    model = get_model()
    model.load_weights(weights_path)                       
    main()
