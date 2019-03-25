#coding=utf-8

import cv2
import random
import os
import numpy as np
from tqdm import tqdm
from osgeo import gdal

img_w = 512  
img_h = 512  

image_sets = ['river_1','river_2','river_3','river_4','river_5','river_6','river_7','river_8','river_9','river_10']

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
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)
        
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)
        
    if np.random.random() < 0.25:
        xb = blur(xb)
    
    if np.random.random() < 0.2:
        xb = add_noise(xb)
        
    return xb,yb

#读图像文件
def read_img(filename):
    dataset=gdal.Open(filename)       #打开文件
    im_width = dataset.RasterXSize    #栅格矩阵的列数
    im_height = dataset.RasterYSize   #栅格矩阵的行数
    im_geotrans = dataset.GetGeoTransform()  #仿射矩阵
    im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵
    del dataset #关闭对象，文件dataset
    return im_proj,im_geotrans,im_data

def write_img(filename,im_proj,im_geotrans,im_data):
        #判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
 
        #判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1,im_data.shape
 
        #创建文件
        driver = gdal.GetDriverByName("GTiff")   #数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
        dataset.SetGeoTransform(im_geotrans)              #写入仿射变换参数
        dataset.SetProjection(im_proj)                    #写入投影
 
        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  #写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])
        del dataset

filepath = 'D:\\RiversTraining\\TwoClasses\\'

def creat_dataset(image_num = 100000, mode = 'original'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        src_proj,src_geotrans,src_img = read_img(filepath + 'src\\' + image_sets[i]+'.tif')  # 3 channels
        src_img = np.array(src_img, dtype = src_img.dtype)
        label_proj,label_geotrans,label_img = read_img(filepath + 'label\\' + image_sets[i]+'_label.tif')
        label_img = np.array(label_img, dtype = label_img.dtype)
        #label_img = cv2.imread(filepath + 'label\\' + image_sets[i]+'_label.tif',cv2.IMREAD_GRAYSCALE)  # single channel
        
        Bands, X_height, X_width = src_img.shape
        Label_height, Label_width = label_img.shape
        #offset_x = int((Label_height - X_height)/2)
        #offset_y = int((Label_width - X_width)/2)
        #label_img = label_img[offset_x: X_height + offset_x, offset_y: X_width + offset_y]
        
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[:,random_height: random_height + img_h, random_width: random_width + img_w]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            if mode == 'augment':
                src_roi,label_roi = data_augment(src_roi,label_roi)
            write_img((filepath + 'training\\src\\%d.tif' % g_count), src_proj,src_geotrans,src_roi)
            write_img((filepath + 'training\\label\\%d.tif' % g_count), src_proj,src_geotrans,label_roi)
            #cv2.imwrite((filepath + 'training\\src\\%d.png' % g_count),src_roi)
            #cv2.imwrite((filepath + 'training\\label\\%d.png' % g_count),label_roi)
            count += 1 
            g_count += 1
            
            
creat_dataset(image_num = 12000)