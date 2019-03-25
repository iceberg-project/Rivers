#coding=utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np  
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation, Input, Dropout
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import img_to_array  
from keras.callbacks import ModelCheckpoint  
from keras import regularizers
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.preprocessing import LabelEncoder  
from PIL import Image
import cv2
import random
import os
from tqdm import tqdm
from osgeo import gdal

from keras import backend as K
import tensorflow as tf
'''
Compatible with tensorflow backend
'''

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#River_net

img_w = 512 
img_h = 512
bands = 8
n_label = 2  
  
classes = [1,255] 
labelencoder = LabelEncoder()  
labelencoder.fit(classes)  
        
def load_img(path, grayscale=False):
    dataset = gdal.Open(path)       #打开文件
    im_width = dataset.RasterXSize    #栅格矩阵的列数
    im_height = dataset.RasterYSize   #栅格矩阵的行数
    im_data = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵
    del dataset #关闭对象，文件dataset
    img = np.array(im_data,dtype = im_data.dtype)   # band first
    if grayscale==False:
        img = np.swapaxes(img,0,1)
        img = np.swapaxes(img,1,2)
    return img

def SegNet():  
    model = Sequential()  
    #encoder  
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(img_w,img_h, 3),padding='same',activation='relu', data_format = 'channels_last', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())  
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    #(128,128)  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(64,64)  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(32,32)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(16,16)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(8,8)  
    #decoder  
    model.add(UpSampling2D(size=(2,2)))  
    #(16,16)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(32,32)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(64,64)  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(128,128)  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(256,256)  
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())  
    model.add(Conv2D(n_label, (1, 1), strides=(1, 1), padding='same', activation='softmax'))  
    #model.add(Reshape((n_label,img_w*img_h)))  
    #axis=1和axis=2互换位置，等同于np.swapaxes(layer,1,2)  
    #model.add(Permute((2,1)))  
    #model.add(Activation('softmax'))  
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
    model.summary()  
    return model

filepath ='D:\\RiversTraining\\TwoClasses\\training\\'

def get_train_val(val_rate = 0.25):
    train_url = []    
    train_set = []
    val_set  = []
    for pic in os.listdir(filepath + 'src'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return train_set,val_set

# data for training  
def generateData(batch_size,data=[]):  
    #print 'generateData...'
    while True:  
        train_data = []  
        train_label = []  
        batch = 0  
        for i in (range(len(data))): 
            url = data[i]
            batch += 1 
            img = load_img(filepath + 'src\\' + url)
            img = img_to_array(img) 
            train_data.append(img)  
            label = load_img(filepath + 'label\\' + url, grayscale=True)            
            label = img_to_array(label).reshape((img_w * img_h,))
            train_label.append(label)
            if batch % batch_size==0: 
                train_data = np.array(train_data)  
                train_label = np.array(train_label).flatten()                
                train_label = labelencoder.transform(train_label)                
                train_label = to_categorical(train_label, num_classes=n_label)  
                train_label = train_label.reshape((batch_size,img_w, img_h,n_label)) 
                yield (train_data,train_label)
                train_data = []  
                train_label = []  
                batch = 0  
 
# data for validation 
def generateValidData(batch_size,data=[]):  
    #print 'generateValidData...'
    while True:  
        valid_data = []  
        valid_label = []  
        batch = 0  
        for i in (range(len(data))):  
            url = data[i]
            batch += 1  
            img = load_img(filepath + 'src\\' + url)
            img = img_to_array(img)  
            valid_data.append(img)  
            label = load_img(filepath + 'label\\' + url, grayscale=True)
            label = img_to_array(label).reshape((img_w * img_h,))  
            valid_label.append(label)  
            if batch % batch_size==0:  
                valid_data = np.array(valid_data)  
                valid_label = np.array(valid_label).flatten()  
                valid_label = labelencoder.transform(valid_label)  
                valid_label = to_categorical(valid_label, num_classes=n_label)  
                valid_label = valid_label.reshape((batch_size,img_w, img_h,n_label))
                yield (valid_data,valid_label)  
                valid_data = []  
                valid_label = []  
                batch = 0  
  
def train(): 
    EPOCHS = 20
    BS = 3
    model = SegNet()
    modelcheck = ModelCheckpoint(filepath + 'img_zeb_tune_4.h5',monitor='val_acc',save_best_only=True,mode='max')  
    callable = [modelcheck]  
    
    train_set,val_set = get_train_val()
    train_numb = len(train_set)  
    valid_numb = len(val_set)  
    print ("the number of train data is",train_numb)  
    print ("the number of val data is",valid_numb)
    
    H = model.fit_generator(generator=generateData(BS,train_set),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,  
                    validation_data=generateValidData(BS,val_set),validation_steps=valid_numb//BS, callbacks=callable,max_q_size=1)  
    model.save(filepath + 'img_zeb4.h5')
    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.title("Training Loss on ZebNet Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(filepath + 'train_zeb_tune4_1.jpg')
    
    plt.figure()
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Accuracy on ZebNet Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(filepath + 'train_zeb_tune4_2.jpg')
    
if __name__=='__main__':  
    train()