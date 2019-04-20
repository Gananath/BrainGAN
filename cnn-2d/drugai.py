import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
import tensorflow.keras.backend as K

#K.set_image_dim_ordering('th')
keras=tf.keras


def dim_chg(x,drop=False):
    if drop==False:
        return x.reshape(x.shape[0],x.shape[1],x.shape[2],1)
    elif drop==True:
        return x[:,:, :, 0]


def mnist_resize(X,w=116,h=116):
    new_img=[]
    for i in X:
        img = resize(i, (w,h), anti_aliasing=True,mode='constant')
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
    
    g_input = (keras.layers.Input(shape=(latent_size,), dtype='float32'))
    bin_switch = (keras.layers.Input(shape=(2,), dtype='float32'))
    
    #x= keras.layers.Dense(256, keras.layers.Activation="relu", keras.layers.Input_dim=latent_size*10)(x)
    #(None, )
    x=keras.layers.concatenate([g_input,bin_switch])
    x= keras.layers.Dense(int(y_dash.shape[1] /4)  * int(y_dash.shape[2]/4) , activation="relu")(x)
    #(None,)
    x= keras.layers.Reshape((int(y_dash.shape[1]/4 ), int(y_dash.shape[2]/4) ))(x)
    #x= keras.layers.BatchNormalization(momentum=0.8)(x)
    x= keras.layers.Dropout(dropout)(x)
    x= keras.layers.UpSampling1D()(x)
    #(None,)
    x= keras.layers.Conv1D(int(y_dash.shape[2] / 3), kernel_size=2, padding="same")(x)
    x= keras.layers.Activation("relu")(x)
    #x= keras.layers.BatchNormalization(momentum=0.8)(x)
    x= keras.layers.Dropout(dropout)(x)
    x= keras.layers.UpSampling1D()(x)
    #(None, )
    x= keras.layers.Conv1D(int(y_dash.shape[2]/ 2), kernel_size=2, padding="same")(x)
    #x= keras.layers.ZeroPadding1D((2,1))(x)
    x= keras.layers.Activation("relu")(x)
    #x= keras.layers.BatchNormalization(momentum=0.8)(x)
    #x= keras.layers.Dropout(dropout)(x)
    #(None, )
    x= keras.layers.Conv1D(int(y_dash.shape[2] / 1), kernel_size=4, padding="same")(x)
    x= keras.layers.Activation("sigmoid")(x)
    
    opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)
    model=keras.models.Model([g_input, bin_switch], x)
    
    print ("keras.layers.Input_shape"+ str(model.input_shape)+"\noutput_shape"+ str(model.output_shape))
    return model



#keras.models.Model.summary()
def Discriminator(y_dash,dropout=0.4,lr=0.00001):
    d_input = keras.layers.Input((int(y_dash.shape[1]), int(y_dash.shape[2])))
    bin_switch = (keras.layers.Input(shape=(1,), dtype='float32'))
    x= keras.layers.Conv1D(input_shape=(int(y_dash.shape[1]), int(y_dash.shape[2])),filters=25,kernel_size=4,padding='same')(d_input)
    x= keras.layers.LeakyReLU()(x)
    x= keras.layers.Dropout(dropout)(x)
    x= keras.layers.MaxPooling1D()(x)
    x= keras.layers.Conv1D(filters=10,
                     kernel_size=4,
                     padding='same')(x)
    x= keras.layers.LeakyReLU()(x)
    x= keras.layers.Dropout(dropout)(x)
    x= keras.layers.MaxPooling1D()(x)
    x= keras.layers.Flatten()(x)
    x= keras.layers.Dense(64)(x)
    x= keras.layers.LeakyReLU()(x)
    x= keras.layers.Dropout(dropout)(x)
    x1= keras.layers.Dense(1,activation='linear',name='fakefind')(x)
    x2= keras.layers.Dense(2,activation='softmax',name='bin_switch')(x)
    model= keras.models.Model(inputs=d_input, outputs=[x1,x2])
    #opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)
   # keras.models.Model.compile(optimizer=opt, loss=[wasserstein_loss,'binary_crossentropy'],              metrics=['accuracy'])
    print ("keras.layers.Input_shape"+ str(model.input_shape)+"\noutput_shape"+ str(model.output_shape))
    return model



def GGenerator(y_dash,lr=0.00001,latent_size=2,dropout=0.4):
    #hybrid CNN2D + CNN1D
    g_inputs = (keras.layers.Input(shape=(latent_size,), dtype='float32'))
    bin_switch = (keras.layers.Input(shape=(2,), dtype='float32'))
    x=keras.layers.concatenate([g_inputs,bin_switch])
    x= keras.layers.Dense(128*7*7 , activation="relu")(x)
    x= keras.layers.LeakyReLU(0.2)(x)    
    x= keras.layers.Reshape((128, 7, 7))(x)
    x= keras.layers.Dropout(dropout)(x)
    x= keras.layers.UpSampling2D(size=(2, 2))(x)
    x= keras.layers.Conv2D(64, kernel_size=(5, 5), padding='same',activation='relu')(x)
    x= keras.layers.LeakyReLU(0.2)(x)
    x= keras.layers.UpSampling2D(size=(2, 2))(x)
    x= keras.layers.Dropout(dropout)(x)
    x= keras.layers.Conv2D(1, kernel_size=(5, 5), padding='same',activation='relu')(x)
    x= keras.layers.Dropout(dropout)(x)
    x= keras.layers.Flatten()(x)
    x= keras.layers.Dropout(dropout)(x)
    x= keras.layers.Dense(116*3,activation='relu')(x)
    x= keras.layers.Reshape((116,3))(x)
    x= keras.layers.Conv1D(y_dash.shape[2],kernel_size=1,padding="same",activation='sigmoid')(x)
    opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)
    model=keras.models.Model([g_inputs, bin_switch], x)
    #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    print("CONV2D+CONV1D")
    print ("input_shape"+ str(model.input_shape)+"\noutput_shape"+ str(model.output_shape))
    return model

def DDiscriminator(y_dash,dropout=0.3,lr=0.00001):
    #hybrid CNN2D + CNN1D
    d_inputs = keras.layers.Input((116, y_dash.shape[2]))
    bin_switch = (keras.layers.Input(shape=(2,), dtype='float32'))
    x= keras.layers.Conv1D(input_shape=(116, y_dash.shape[2]),filters=y_dash.shape[2]-2, kernel_size=4,padding='same',activation='relu')(d_inputs)
    x= keras.layers.Reshape((1,116,y_dash.shape[2]-2))(x)
    x= keras.layers.LeakyReLU(0.2)(x)
    x= keras.layers.Dropout(dropout)(x)
    x= keras.layers.Conv2D(50, kernel_size=(5, 5), strides=(2, 2), padding='same',activation='relu')(x)
    x= keras.layers.LeakyReLU(0.2)(x)
    x= keras.layers.Dropout(dropout)(x)
    x= keras.layers.Flatten()(x)
    x= keras.layers.Dense(512,activation='relu')(x)
    x= keras.layers.Dropout(dropout)(x)
    x= keras.layers.Dense(10)(x)
    x1= keras.layers.Dense(1,activation='linear',name='fakefind')(x)
    x2= keras.layers.Dense(2,activation='softmax',name='bin_switch')(x)    
    model = keras.models.Model(inputs=d_inputs, outputs=[x1,x2])
    print("CONV2D+CONV1D")
    print ("input_shape"+ str(model.input_shape)+"\noutput_shape"+ str(model.output_shape))
    return model


def Generator2D(lr=0.00001,latent_size=2,dropout=0.5):
    #hybrid CNN2D 
    g_inputs = (keras.layers.Input(shape=(latent_size,), dtype='float32'))
    bin_switch = (keras.layers.Input(shape=(2,), dtype='float32'))
    x=keras.layers.concatenate([g_inputs,bin_switch])
    x= keras.layers.Dense(29*6*1 , activation="relu")(x)
    #x= keras.layers.LeakyReLU(0.2)(x)
    x= keras.layers.BatchNormalization(momentum=0.8)(x)
    x= keras.layers.Reshape((29, 6, 1))(x)
    x= keras.layers.Dropout(dropout)(x)
    x= keras.layers.UpSampling2D(size=(2, 2))(x)
    x= keras.layers.Conv2D(1, kernel_size=(5, 5), padding='same',activation='relu', data_format="channels_last")(x)
    #x= keras.layers.LeakyReLU(0.2)(x)
    x= keras.layers.BatchNormalization(momentum=0.8)(x)
    x= keras.layers.Dropout(dropout)(x)
    x= keras.layers.UpSampling2D(size=(2, 2))(x)
    x= keras.layers.Conv2D(1, kernel_size=(5, 5), padding='same',activation='sigmoid',data_format="channels_last")(x)
    opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)
    model=keras.models.Model([g_inputs, bin_switch], x)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    print("CONV2D")
    print ("input_shape"+ str(model.input_shape)+"\noutput_shape"+ str(model.output_shape))
    return model

def Discriminator2D(dropout=0.3,lr=0.00001):
    #hybrid CNN2D 
    d_inputs = keras.layers.Input((116, 24,1))
    bin_switch = (keras.layers.Input(shape=(2,), dtype='float32'))
    x= keras.layers.Conv2D(50,input_shape=(116, 24,1), kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu',data_format="channels_last")(d_inputs)
    #x= keras.layers.LeakyReLU(0.2)(x)
    x= keras.layers.BatchNormalization(momentum=0.8)(x)
    x= keras.layers.Dropout(dropout)(x)
    x= keras.layers.Conv2D(50, kernel_size=(5, 5), strides=(2, 2), padding='same',activation='relu',data_format="channels_last")(x)
    #x= keras.layers.LeakyReLU(0.2)(x)
    x= keras.layers.BatchNormalization(momentum=0.8)(x)
    x= keras.layers.Dropout(dropout)(x)
    x= keras.layers.Flatten()(x)
    x= keras.layers.Dense(128,activation='relu')(x)
    x= keras.layers.Dropout(dropout)(x)
    x= keras.layers.Dense(10)(x)
    x1= keras.layers.Dense(1,activation='linear',name='fakefind')(x)
    x2= keras.layers.Dense(2,activation='softmax',name='bin_switch')(x)
    opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)
    model = keras.models.Model(inputs=d_inputs, outputs=[x1,x2])
    model.compile(loss=[wasserstein_loss,'categorical_crossentropy'], optimizer=opt)
    print("CONV2D")
    print ("input_shape"+ str(model.input_shape)+"\noutput_shape"+ str(model.output_shape))
    return model
