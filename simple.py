import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation,Reshape
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import os
import pickle
import numpy as np
import dataset
import cv2
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
from keras.utils import to_categorical
from sklearn import preprocessing
import tensorflow as tf

x0=np.random.randint(6, size=(16, 1))
x1=np.random.randint(6, size=(16, 1))




#Prepare input data
l = 1
A=1*l
fy1 = np.array([1,1,1,-1,-1,-1])
fy2 = np.array([-1,1,-1,1,-1,1])
#print(np.transpose(fy1).reshape(6,1)*fy2.reshape(1,6))
py1y2= (l*np.transpose(fy1).reshape(6,1)*fy2.reshape(1,6)) + A*np.ones((6,6)).astype(int)
print(py1y2)
#print(np.shape(py1y2))
#py1y2=preprocessing.normalize(py1y2,norm='l2')
#print(py1y2)
n = to_categorical(py1y2)
#print(n)


data = py1y2
#print(np.shape(data))
#print("Complete reading input data. Will Now print a snippet of it")

#total_pixels = img_size_w * img_size_h * 3
translator_factor = 2
translator_layer_size = 500
middle_factor = 2
middle_layer_size = 250
batch_size = 32
num_classes = 10
epochs = 100

x_train, _, _, _ = data[:4]
x_test, _, _, _ = data[2:6]
#print(np.shape(x_train),np.shape(x_test))
x0_train = data[:2]
#print(x0_train)
x0_test = x_test[4:5]
x1_train = x_train[:2]
x1_test = x_test[4:5]

x0_val = data[:4]
x0_test = x0_test[4:]

x1_val = data[:4]
x1_test = x1_test[4:]

#print("validation data: {0} \ntest data: {1}".format(x0_val.shape, x_test.shape))

# Model Y0 ->  X0

input_img = Input(shape=(6,1,1))
#print("inp : ",input_img)0
x0 = Conv2D(64, (3, 3), padding='same')(input_img)
#print("x0",x0)
x0 = BatchNormalization()(x0)
x0 = Activation('relu')(x0)
x0 = MaxPooling2D((2, 2), padding='same')(x0)
x0 = Conv2D(32, (3, 3), padding='same')(x0)
x0 = BatchNormalization()(x0)
x0 = Activation('relu')(x0)
x0 = MaxPooling2D((2, 2), padding='same')(x0)
x0 = Conv2D(16, (3, 3), padding='same')(x0)
x0 = BatchNormalization()(x0)
x0 = Activation('relu')(x0)
#print(x0._keras_shape)
x0=Reshape((-1,2,1))(x0)
encoded_x0 = MaxPooling2D((2, 2), padding='same')(x0)
#print("Encoded",encoded_x0._keras_shape)


model_x0 = Model(input_img,encoded_x0)
w = model_x0.get_weights()

#Model Y1 -> X1
x1 = Conv2D(64, (3, 3), padding='same')(input_img)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = MaxPooling2D((2, 2), padding='same')(x1)
x1 = Conv2D(32, (3, 3), padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = MaxPooling2D((2, 2), padding='same')(x1)
x1 = Conv2D(16, (3, 3), padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1=Reshape((-1,2,1))(x1)

encoded_x1 = MaxPooling2D((2, 2), padding='same')(x1)

model_x1 = Model(input_img,encoded_x1)
model_x1.set_weights(w)

x0_train = x0_train.reshape((len(x0_train), np.prod(x0_train.shape[1:]),1,1))
#print(np.shape(x0_train))
x0_val = x0_val.reshape((len(x0_val), np.prod(x0_val.shape[1:]),1,1))

#print(np.shape(x0_val))
def loss_function(a, b):
    a1 = tf.reshape(tensor=encoded_x0, shape=(-1, 8))
    a2=tf.reshape(tensor=encoded_x1,shape=(-1,8))
    dot_a1a2= tf.matmul(a1,a2,transpose_b=True)
    b1=tf.matmul(a1,a1,transpose_a=True)
    #print(tf.shape(b1))
    b2=tf.matmul(a2,a2,transpose_a=True)
    #print(tf.shape(b2))
    h = -dot_a1a2/8 +(b1*b2)/2
    #print(dot_a1a2)
    #sess = K.get_session()
    #b1 = sess.run(a1)
    #b1=tensor_to_array(a1)
    #b2=(type(tf.Session().run(tf.constant(a2))))
    #print(b1,b2)
    #a1=K.reshape(encoded_x0,[8])
    #b1=K.reshape(x1,(-
    return h
model_x0.compile(optimizer='Adam', loss=loss_function)
history_x0 = model_x0.fit(x0_train, np.ones(np.shape(x0_train)),batch_size=batch_size,epochs=epochs,validation_data=(x0_val,np.ones(np.shape(x0_val)) ),shuffle=True)

model_x1.compile(optimizer='Adam', loss=loss_function)
history_x1 = model_x1.fit(x0_train, np.ones(np.shape(x0_train)),batch_size=batch_size,epochs=epochs,validation_data=(x0_val,np.ones(np.shape(x0_val)) ),shuffle=True)

z = model_x0.predict(x0_train)
print(z)