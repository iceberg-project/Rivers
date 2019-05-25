#coding=utf-8

import cv2
import random
import os
import numpy as np
from tqdm import tqdm
from osgeo import gdal


#Ten overlapped worldview images with labels
image_sets = ['river_1','river_2','river_3','river_4','river_5','river_6','river_7','river_8','river_9','river_10']

#image size in the training set
img_w = 512  
img_h = 512  


#####################image augmentation
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb,yb,angle):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb

def blur(img):
    img = cv2.blur(img, (3, 3));
    return img

def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb,yb):
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,90)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,180)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：flip over y axis
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb,yb

#read worldview image
def read_img(filename):
    dataset=gdal.Open(filename)       #open the file
    im_width = dataset.RasterXSize    #the number of column
    im_height = dataset.RasterYSize   #the number of row
    im_geotrans = dataset.GetGeoTransform()  #affine matrix
    im_proj = dataset.GetProjection() #the projection information
    im_data = dataset.ReadAsArray(0,0,im_width,im_height) #convert image data into array
    del dataset #close dataset object
    return im_proj,im_geotrans,im_data

#write image into disk.
def write_img(filename,im_proj,im_geotrans,im_data):
        #decide datatype of the image
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
 
        #decide the dimension of the data (1 dimension or 3 dimension)
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1,im_data.shape
 
        #create file
        driver = gdal.GetDriverByName("GTiff")   #get the geotiff driver
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
        dataset.SetGeoTransform(im_geotrans)              #set affine matrix
        dataset.SetProjection(im_proj)                    #set the projection
 
        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  #write the image data into the file
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])
        del dataset

#local base folder for the project.  The structure of this project is as following
#  \Base_folder
#       \scr
#       \label
#       \training
#            \src
#            \label
filepath = 'D:\\RiversTraining\\TwoClasses\\'


#create training sets from ten overlapped images. image_num means the number of images in training sets. Mode can be original or augment
def creat_dataset(image_num = 100000, mode = 'original'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        src_proj,src_geotrans,src_img = read_img(filepath + 'src\\' + image_sets[i]+'.tif')  # read worldview images
        src_img = np.array(src_img, dtype = src_img.dtype)
        label_proj,label_geotrans,label_img = read_img(filepath + 'label\\' + image_sets[i]+'_label.tif') #read labels
        label_img = np.array(label_img, dtype = label_img.dtype)

        Bands, X_height, X_width = src_img.shape
        Label_height, Label_width = label_img.shape

        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[:,random_height: random_height + img_h, random_width: random_width + img_w]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            if mode == 'augment':
                src_roi,label_roi = data_augment(src_roi,label_roi)
            write_img((filepath + 'training\\src\\%d.tif' % g_count), src_proj,src_geotrans,src_roi)
            write_img((filepath + 'training\\label\\%d.tif' % g_count), src_proj,src_geotrans,label_roi)
            count += 1 
            g_count += 1

creat_dataset(image_num = 12000)
