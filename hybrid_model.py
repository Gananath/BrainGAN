#Sample Hybrid model Conv2D+Conv1D
'''
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
'''
import numpy as np
#from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.layers import Input,concatenate
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam

from keras.layers.convolutional import Conv2D, UpSampling2D,Conv1D
from keras import backend as K
from keras import initializers

K.set_image_dim_ordering('th')



def Generator(y_dash,lr=0.00001,latent_size=2,dropout=0.5):
    #hybrid CNN2D + CNN1D
    y_dash=0    
    g_inputs = (Input(shape=(latent_size,), dtype='float32'))
    bin_switch = (Input(shape=(2,), dtype='float32'))
    x=concatenate([g_inputs,bin_switch])
    x= Dense(128*7*7 , activation="relu")(x)
    x= LeakyReLU(0.2)(x)    
    x= Reshape((128, 7, 7))(x)
    x= Dropout(dropout)(x)
    x= UpSampling2D(size=(2, 2))(x)
    x= Conv2D(64, kernel_size=(5, 5), padding='same',activation='relu')(x)
    x= LeakyReLU(0.2)(x)
    x= UpSampling2D(size=(2, 2))(x)
    x= Dropout(dropout)(x)
    x= Conv2D(1, kernel_size=(5, 5), padding='same',activation='relu')(x)
    x= Dropout(dropout)(x)
    x= Flatten()(x)
    x= Dense(116*3,activation='relu')(x)
    x= Dropout(dropout)(x)
    x= Reshape((116,3))(x)
    x= Conv1D(28,kernel_size=1,padding="same",activation='sigmoid')(x)
    opt = Adam(lr, beta_1=0.5, beta_2=0.9)
    model=Model([g_inputs, bin_switch], x)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    print("CONV2D+CONV1D")
    print ("input_shape"+ str(model.input_shape)+"\noutput_shape"+ str(model.output_shape))
    return model

def Discriminator(y_dash,dropout=0.5,lr=0.00001):
    #hybrid CNN2D + CNN1D
    y_dash=0
    d_inputs = Input((116, 28))
    bin_switch = (Input(shape=(2,), dtype='float32'))
    x= Conv1D(input_shape=(116, 28),nb_filter=25, filter_length=4,border_mode='same',activation='relu')(d_inputs)
    x= Reshape((1,116,25))(x)
    x= LeakyReLU(0.2)(x)
    x= Dropout(dropout)(x)
    x= Conv2D(50, kernel_size=(5, 5), strides=(2, 2), padding='same',activation='relu')(x)
    x= LeakyReLU(0.2)(x)
    x= Dropout(dropout)(x)
    x= Flatten()(x)
    x= Dense(512,activation='relu')(x)
    x= Dropout(dropout)(x)
    x= Dense(10)(x)
    x1= Dense(1,activation='linear',name='fakefind')(x)
    x2= Dense(2,activation='softmax',name='bin_switch')(x)    
    model = Model(input=d_inputs, output=[x1,x2])
    print("CONV2D+CONV1D")
    print ("input_shape"+ str(model.input_shape)+"\noutput_shape"+ str(model.output_shape))
    return model

Generator(1)
Discriminator(1)


