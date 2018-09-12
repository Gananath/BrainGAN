import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape,Input, Merge,concatenate
from keras.layers import Conv1D, UpSampling1D, MaxPooling1D, ZeroPadding1D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.utils import plot_model
import matplotlib.pyplot as plt
from skimage.transform import resize
import keras.backend as K

def mnist_resize(X,w=116,h=116):
    new_img=[]
    for i in X:
        img = resize(i, (w,h), anti_aliasing=True)
        new_img.append(img)
    return np.array(new_img)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)
# time step addtition to feature

def plot_mnist(X_test):
    plt.subplot(221)
    plt.imshow(X_test[0], cmap=plt.get_cmap('gray'))
    plt.subplot(222)
    plt.imshow(X_test[1], cmap=plt.get_cmap('gray'))
    plt.subplot(223)
    plt.imshow(X_test[2], cmap=plt.get_cmap('gray'))
    plt.subplot(224)
    plt.imshow(X_test[3], cmap=plt.get_cmap('gray'))
    # show the plot
    plt.show()    

def dimX(x, ts):
    x = np.asarray(x)
    newX = []
    for i, c in enumerate(x):
        newX.append([])
        for j in range(ts):
            newX[i].append(c)
    return np.array(newX)


def train_test_split(X, y, percentage=0.75):
    p = int(len(X) * percentage)
    X_train = X[0:p]
    Y_train = y[0:p]
    X_test = X[p:]
    Y_test = y[p:]
    return X_train, X_test, Y_train, Y_test


# time step addtition to target
def dimY(Y, ts, char_idx, chars):
    temp = np.zeros((len(Y), ts, len(chars)), dtype=np.bool)
    for i, c in enumerate(Y):
        for j, s in enumerate(c):
            # print i, j, s
            temp[i, j, char_idx[s]] = 1
    return np.array(temp)


# prediction of argmax


def prediction(preds):
    y_pred = []
    for i, c in enumerate(preds):
        y_pred.append([])
        for j in c:
            y_pred[i].append(np.argmax(j))
    return np.array(y_pred)
# sequence to text conversion


def seq_txt(y_pred, idx_char):
    newY = []
    for i, c in enumerate(y_pred):
        newY.append([])
        for j in c:
            newY[i].append(idx_char[j])

    return np.array(newY)

# joined smiles output


def smiles_output(s):
    smiles = np.array([])
    for i in s:
        #j = ''.join(str(k.encode('utf-8')) for k in i)
        j = ''.join(str(k) for k in i)
        smiles = np.append(smiles, j)
    return smiles


def Generator(y_dash,lr=0.00001,latent_size=2,dropout=0.4):
    dropout=dropout
    g_inputs = (Input(shape=(latent_size,), dtype='float32'))
    bin_switch = (Input(shape=(1,), dtype='float32'))

    #x= Dense(256, activation="relu", input_dim=latent_size*10)(x)
    #(None, )
    x=concatenate([g_inputs,bin_switch])
    x= Dense((y_dash.shape[1] /4)  * (y_dash.shape[2]/4) , activation="relu")(x)
    #(None,)
    x= Reshape(((y_dash.shape[1]/4 ), y_dash.shape[2]/4 ))(x)
    #x= BatchNormalization(momentum=0.8)(x)
    x= Dropout(dropout)(x)
    x= UpSampling1D()(x)
    #(None,)
    x= Conv1D(y_dash.shape[2] / 3, kernel_size=2, padding="same")(x)
    x= Activation("relu")(x)
    #x= BatchNormalization(momentum=0.8)(x)
    x= Dropout(dropout)(x)
    x= UpSampling1D()(x)
    #(None, )
    x= Conv1D(y_dash.shape[2] / 2, kernel_size=2, padding="same")(x)
    #x= ZeroPadding1D((2,1))(x)
    x= Activation("relu")(x)
    #x= BatchNormalization(momentum=0.8)(x)
    x= Dropout(dropout)(x)
    #(None, )
    x= Conv1D(y_dash.shape[2] / 1, kernel_size=4, padding="same")(x)
    x= Activation("softmax")(x)
    
    opt = Adam(lr, beta_1=0.5, beta_2=0.9)
    model=Model([g_inputs, bin_switch], x)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    print ("input_shape"+ str(model.input_shape)+"\noutput_shape"+ str(model.output_shape))
    return model



#model.summary()
def Discriminator(y_dash,dropout=0.4,lr=0.00001):
    d_inputs = Input((y_dash.shape[1], y_dash.shape[2]))
    bin_switch = (Input(shape=(1,), dtype='float32'))
    x= Conv1D(input_shape=(y_dash.shape[1], y_dash.shape[2]),
                     nb_filter=25,
                     filter_length=4,
                     border_mode='same')(d_inputs)
    x= LeakyReLU()(x)
    x= Dropout(dropout)(x)
    x= MaxPooling1D()(x)
    x= Conv1D(nb_filter=10,
                     filter_length=4,
                     border_mode='same')(x)
    x= LeakyReLU()(x)
    x= Dropout(dropout)(x)
    x= MaxPooling1D()(x)
    x= Flatten()(x)
    x= Dense(64)(x)
    x= LeakyReLU()(x)
    x= Dropout(dropout)(x)
    x1= Dense(1,activation='linear',name='fakefind')(x)
    x2= Dense(1,activation='softmax',name='bin_switch')(x)    
    model = Model(input=d_inputs, output=[x1,x2])
    #opt = Adam(lr, beta_1=0.5, beta_2=0.9)
    
   # model.compile(optimizer=opt, loss=[wasserstein_loss,'binary_crossentropy'],              metrics=['accuracy'])
    print ("input_shape"+ str(model.input_shape)+"\noutput_shape"+ str(model.output_shape))
    return model


