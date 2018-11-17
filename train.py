import  numpy  as  np
from  keras import  layers
from  keras.layers  import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras import optimizers
from keras import models
from keras import callbacks

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import os
import cv2 as cv
import numpy as np
import shutil
import glob
from sklearn.model_selection import train_test_split


def create_input(imagecnt, image_size):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    x_arr_ori = np.zeros((imagecnt, image_size, image_size , 3))
    y_arr_ori = np.zeros((imagecnt, 1))

    for i,image in enumerate(images):
        img = cv.imread(image)
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        rgbimg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        face = face_cascade.detectMultiScale(grey,1.2,2)
        x,y,w,h = face[0]
        trim = rgbimg[y:y+h,x:x+w]
        trim = cv.resize(trim, (image_size,image_size))  
        x_arr_ori[i,:,:,:] = trim.astype('float32')
        y_arr_ori[i] = int(os.path.basename(image)[:2]) -1
    
    _y_arr_ori = y_arr_ori.reshape(y_arr_ori.shape[1], y_arr_ori.shape[0])
    y_onehot = to_categorical(_y_arr_ori)[0]

    return   x_arr_ori, y_onehot

def add_sideflip(X_train,Y_train):
    flp_arr = np.empty(X_train.shape)

    for i,arr in enumerate(X_train):
        flp_arr[i] = np.fliplr(arr)

    X_train_addflip = np.concatenate((X_train, flp_arr), axis = 0)
    Y_train_addflip = np.concatenate((Y_train, Y_train), axis = 0)

    return X_train_addflip,Y_train_addflip

def normalize(X):
    X_norm = X/255.
    return X_norm



def define_model(image_size,):
    input_tensor = Input(shape=(image_size, image_size, 3))
    vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    for layer in vgg16_model.layers:
            layer.trainable = False
        
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(1024))
    top_model.add(Activation("relu"))
    top_model.add(Dropout(0.51))
    top_model.add(Dense(Y_train.shape[1]))
    top_model.add(Activation("softmax"))
    model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))

    return model

if __name__ == '__main__':
    imagedir = './images/'
    images = glob.glob(imagedir + '*.jpg')
    imagecnt = len(images)
    image_size = 224

    x_arr_ori, y_onehot = create_input(imagecnt, image_size)

    X_train, X_test, Y_train, Y_test = train_test_split(x_arr_ori, y_onehot, test_size=0.1)

    X_train,Y_train = add_sideflip(X_train, Y_train)
    
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    print('X_train size : ' + str(X_train.shape))
    print('Y_train size : ' + str(Y_train.shape))
    print('X_test size : ' + str(X_test.shape))
    print('Y_test size : ' + str(Y_test.shape))

    model = define_model(image_size)

    epochs = 300
    batch_size=128
    sgd = optimizers.SGD(lr=0.001,momentum=0.9,nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics = ["accuracy"])
    filename = './model/bestmodel_20180818' + '_e' + str(epochs) + '_bs' + str(batch_size) + '_do051'
    cp_cb = callbacks.ModelCheckpoint(filepath = filename, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    stack  = model.fit(x=X_train, y= Y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_test,Y_test),callbacks=[cp_cb])

