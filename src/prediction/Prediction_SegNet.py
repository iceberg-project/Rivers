import cv2
import random
import numpy as np
import os
import argparse
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from osgeo import gdal

#image set used to test model
TEST_SET = ['12JUL181553195.tif','12JUL181553219.tif','12JUL211543217.tif','12JUL211543228.tif','12JUL211543239.tif']

#small patch size
image_size = 512

#local base directory for test data
base_directory = 'D:\\RiversTraining\\TwoClasses\\'

#label encoder to generate similar prediction images as training labels
classes = [1,255] 
labelencoder = LabelEncoder()  
labelencoder.fit(classes)

# def args_parse():
# # construct the argument parse and parse the arguments
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-m", "--model", required=True,
#         help="path to trained model model")
#     ap.add_argument("-s", "--stride", required=False,
#         help="crop slide stride", type=int, default=image_size)
#     args = vars(ap.parse_args())
#     return args

#load images for prediction
def load_img(path, grayscale=False):
    dataset = gdal.Open(path)       #open the file
    im_width = dataset.RasterXSize    #the number of column for the image
    im_height = dataset.RasterYSize   #the number of row for the image
    im_geotrans = dataset.GetGeoTransform()  #Affine Matrix
    im_proj = dataset.GetProjection() #Projection information for the image
    im_data = dataset.ReadAsArray(0,0,im_width,im_height) #convert image into array, which corresponds to image matrix.
    del dataset #close dataset object
    img = np.array(im_data,dtype=im_data.dtype) 
    if grayscale==False: #if not grayscale image, change the order of the axis
        img = np.swapaxes(img,0,1)
        img = np.swapaxes(img,1,2)
    return im_geotrans,im_proj, img

#write predictions into local disks.
def write_img(filename,im_proj,im_geotrans,im_data):
        #decide which the datatype of the image is 
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
 
        print(im_data.shape)
        #decide the dimension of the image (worldview image or label image)
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1,im_data.shape
 
        #创建文件
        driver = gdal.GetDriverByName("GTiff")   #create a geotiff driver
        print(filename, im_width, im_height, im_bands, datatype)
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
        dataset.SetGeoTransform(im_geotrans)              #Set affine transformation parameters
        dataset.SetProjection(im_proj)                    #set projection information
 
        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  #write ths image data
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])
        del dataset
    
def predict():
    # load the trained convolutional neural network
    model = load_model(base_directory+'training\\SegNet.h5')
    # stride for crop the original images
    stride = int(image_size/2)
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        #load the image
        geotrans, proj, image = load_img(base_directory+'src\\' + path)
        h,w,bands = image.shape

        #compute the image size after padding
        padding_h = ((h - image_size)//stride + 2) * stride + image_size 
        padding_w = ((w - image_size)//stride + 2) * stride + image_size 
        padding_img = np.zeros((padding_h,padding_w,bands),dtype=image.dtype)
        padding_img[stride//2:h+stride//2,stride//2:w+stride//2,:] = image[:,:,:]
        padding_img[0:stride//2,stride//2:w+stride//2,:] = image[0:stride//2,:,:]
        padding_img[h+stride//2:padding_h,stride//2:w+stride//2,:] = image[h-(padding_h-h-stride//2):h,:,:]
        padding_img[:,0:stride//2,:] = padding_img[:,stride//2:stride,:]
        padding_img[:,w+stride//2:padding_w,:] = padding_img[:,w+stride//2-(padding_w-w-stride//2):w+stride//2,:]
        padding_img = img_to_array(padding_img)

        # create output numpy matrix for labels
        mask_whole = np.zeros((padding_h,padding_w),dtype=np.float64)

        for i in range((padding_h-image_size)//stride+1):
            for j in range((padding_w-image_size)//stride+1):
                #crop the whole image to get 512*512 patches
                crop = padding_img[i*stride:i*stride+image_size,j*stride:j*stride+image_size,:]
                ch,cw,_ = crop.shape
                crop = np.expand_dims(crop, axis=0)

                #use the model to make prediction about the input small patches.
                pred = model.predict(crop,verbose=2)
                pred = np.argmax(pred, axis=3).flatten()

                #decode the output class into label values
                pred = labelencoder.inverse_transform(pred)
                pred = pred.reshape((image_size,image_size))

                #mosaic the small patch into the whole label image.
                mask_whole[stride//2+i*stride:i*stride+image_size-stride//2,j*stride+stride//2:j*stride+image_size-stride//2] = pred[stride//2:image_size-stride//2,stride//2:image_size-stride//2]
        #write the label for the whole image into  disk.
        write_img(base_directory+'prediction\\'+TEST_SET[n]+'_label_zeb_tune4.tif',proj,geotrans, mask_whole[stride//2:h+stride//2,stride//2:w+stride//2])

if __name__ == '__main__':
    predict()
