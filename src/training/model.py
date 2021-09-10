import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as k
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.utils import plot_model

# Custom IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def unet(n_classes=1, im_sz=256, n_channels=1, n_filters_start=32, growth_factor=2, upconv=True):
    n_filters = n_filters_start
    inputs = Input((im_sz, im_sz, n_channels))
    conv1 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    n_filters *= growth_factor
    conv2 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    n_filters *= growth_factor
    conv3 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    n_filters *= growth_factor
    conv4 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(conv4)
    pool4 = Dropout(0.25)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(pool4)

    n_filters *= growth_factor
    conv5 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(conv5)
    pool5 = Dropout(0.25)(conv5)

    n_filters //= growth_factor
    if upconv:
        up6 = concatenate([Conv2DTranspose(n_filters, (2, 2), activation = 'relu', strides=(2, 2), padding='same')(conv5), conv4])
    else:
        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(up6)
    conv6 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(conv6)

    n_filters //= growth_factor
    if upconv:
        up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), activation = 'relu',  strides=(2, 2), padding='same')(conv6), conv3])
    else:
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(conv7)

    n_filters //= growth_factor
    if upconv:
        up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), activation = 'relu', strides=(2, 2), padding='same')(conv7), conv2])
    else:
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(conv8)

    n_filters //= growth_factor
    if upconv:
        up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), activation = 'relu', strides=(2, 2), padding='same')(conv8), conv1])
    else:
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(up9)
    conv9 = Conv2D(n_filters, (3, 3), activation = 'relu', padding = 'same')(conv9)

    conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer=Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy',f1_m,precision_m, recall_m])
    model.summary()

    return model

