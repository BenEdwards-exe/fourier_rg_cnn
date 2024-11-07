# The models were taken from https://github.com/BurgerBecker/rg-benchmarker and cross-checked with their original papers

import tensorflow as tf
from keras import initializers, regularizers
from keras.models import Sequential
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization

# ---------------------------------------
# Regularised Models
# ---------------------------------------
def MCRGNet_regularised(input_shape, n_classes, random_seed):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3,3),kernel_regularizer=regularizers.l2(0.01), strides = 2, activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_seed))
    model.add(Conv2D(8, (3, 3), strides = 2,kernel_regularizer=regularizers.l2(0.01), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_seed))
    model.add(Conv2D(16, (3, 3), strides = 2,kernel_regularizer=regularizers.l2(0.01), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_seed))
    model.add(Conv2D(16, (3, 3), strides = 2,kernel_regularizer=regularizers.l2(0.01), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_seed))
    model.add(Conv2D(32, (3, 3), strides = 2,kernel_regularizer=regularizers.l2(0.01), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_seed))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_seed))
    model.add(Dense(n_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_seed),bias_initializer='zeros'))
    return model


def FR_Deep_regularised(input_shape, n_classes, random_seed):
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(11,11),strides=1,kernel_regularizer=regularizers.l2(0.01), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='valid'))
    model.add(Dropout(0.25,seed=random_seed))
    model.add(Conv2D(16, (5, 5),strides=1,padding='same',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=3, padding='valid'))
    model.add(Dropout(0.25,seed=random_seed))
    model.add(Conv2D(24, (3, 3), strides=1, padding='same',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25,seed=random_seed))
    model.add(Conv2D(24, (3, 3), strides=1,padding='same',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25,seed=random_seed))
    model.add(Conv2D(16, (3, 3), strides=1,padding='same',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(5,5), strides=5, padding='valid'))
    model.add(Dropout(0.25,seed=random_seed))
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_seed), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_seed))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_seed), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_seed))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_seed), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_seed))
    model.add(Dense(n_classes, kernel_initializer=initializers.glorot_uniform(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def RG_ZOO_regularised(input_shape, n_classes, random_seed):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(8, 8), strides=3, padding='same', kernel_regularizer=regularizers.l2(0.01), input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (7, 7), strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25, seed=random_seed))
    model.add(Conv2D(64, (2, 2),strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),name='final_output'))
    model.add(Dropout(0.25, seed=random_seed))
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_seed))
    model.add(Dense(1024, kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_seed))
    model.add(Dense(n_classes, kernel_initializer=initializers.glorot_uniform(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def Toothless_regularised(input_shape, n_classes, random_seed):
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11,11), padding='valid', kernel_regularizer=regularizers.l2(0.01), strides=4,input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=2,padding='valid'))
    model.add(Dropout(0.25, seed=random_seed))
    model.add(Conv2D(256, (5, 5), padding='same', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    model.add(Dropout(0.25, seed=random_seed))
    model.add(Conv2D(384, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Conv2D(384, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    model.add(Dropout(0.25, seed=random_seed))
    model.add(Flatten())
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_seed))
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_seed))
    model.add(Dense(n_classes, kernel_initializer=initializers.glorot_uniform(seed=random_seed),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model







# ---------------------------------------
# No added regularisation
# ---------------------------------------

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



def MCRGNet(input_shape=(150,150,1),num_classes=3,random_state=42):
    
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3,3), strides = 2, activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(8, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.25, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))

    return model
