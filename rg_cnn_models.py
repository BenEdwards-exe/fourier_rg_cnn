# Author: B. Becker
# Title: CNN architecture comparison for radio galaxy classification
# https://github.com/BurgerBecker/rg-benchmarker

import tensorflow as tf
from keras import initializers
from keras.models import Sequential
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization


def Toothless(input_shape=(150, 150, 1), output_classes=3, random_state=42):
  
    model = Sequential(name="toothless_"+str(output_classes)+"_class")
    model.add(Conv2D(96, kernel_size=(11,11), padding='valid', strides=4,input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=2,padding='valid'))
    model.add(Conv2D(256, (5, 5), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    model.add(Flatten())
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))  
    model.add(Dense(output_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))


    return model




def FR_DEEP(input_shape,num_classes,random_state=42):
    # Transfer learning for radio galaxy classification
    # Primary Author: Hongming Tang
    # https://github.com/HongmingTang060313/FR-DEEP
    # https://academic.oup.com/mnras/article/488/3/3358/5538844
    # https://arxiv.org/pdf/1903.11921.pdf
    # Architecture description: Page 7
    # 25 July 2019
    # Notes: This is the original architecture
    
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(11,11),strides=1,padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    model.add(Conv2D(16, (5, 5),strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),strides=3))
    model.add(Conv2D(24, (3, 3),strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(24, (3, 3),strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3),strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(5,5),strides=5))
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model



def RG_ZOO(input_shape,num_classes,random_state=42):
    # Title: Radio Galaxy Zoo: compact and extended radio source classification with deep learning
    # https://academic.oup.com/mnras/article/476/1/246/4826039
    # Primary Author: Vesna Lukic
    # Publication date: 26 Jan 2018

    # Notes: We assume that the padding method used is same padding

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(8, 8), strides=3, padding='same', input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (7, 7), strides=2, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, (2, 2),strides=1, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),name='final_output'))
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(1024, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

