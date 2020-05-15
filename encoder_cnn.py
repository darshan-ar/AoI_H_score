import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
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

def loss_function(x0,x1):
    print("1: ",x0,"2: ",x1)
    return K.sum(K.log(x0) - K.log(x1))

#Prepare input data
train_path='training_data'
classes = ['video_frames']
print(classes)
num_classes = len(classes)

# 10% of the data will automatically be used for validation
validation_size = 0.1
img = cv2.imread('C:\\Users\\darsh\\PycharmProjects\\final\\training_data\\video_frames\\video_frames 0001.jpg')
img_size_w,img_size_h,_ = img.shape
num_channels = 3
sample_size = 1926

data = dataset.read_train_sets(train_path, 400, ['video_frames'], validation_size=validation_size, sample_size=sample_size)
print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

total_pixels = img_size_w * img_size_h * 3
translator_factor = 2
translator_layer_size = 500
middle_factor = 2
middle_layer_size = 250
batch_size = 32
num_classes = 10
epochs = 100

x_train, _, _, _ = data.train.next_batch(1776)
x_test, _, _, _ = data.valid.next_batch(150)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x0_train = x_train[:1774]
x0_test = x_test[:148]
x1_train = x_train[1:1775]
x1_test = x_test[1:149]

x0_val = x0_test[:80]
x0_test = x0_test[80:]

x1_val = x1_test[:80]
x1_test = x1_test[80:]

#print("validation data: {0} \ntest data: {1}".format(x0_val.shape, x_test.shape))

# Model Y0 ->  X0

input_img = Input(shape=(400, 400, 3))
#print("inp : ",input_img)
x0 = Conv2D(64, (3, 3), padding='same')(input_img)
print("x0",x0)
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
encoded_x0 = MaxPooling2D((2, 2), padding='same')(x0)
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
encoded_x1 = MaxPooling2D((2, 2), padding='same')(x1)
model_x1 = Model(input_img,encoded_x1)
model_x1.set_weights(w)

model_x = keras.layers.concatenate([encoded_x0, ])

model_x0.compile(optimizer='Adam', loss=loss_function)

history = model_x0.fit(x0_train, x1_train,batch_size=batch_size,epochs=epochs,validation_data=(x0_val, x1_val),shuffle=True)

score = model.evaluate(x_test, x_test, verbose=1)
print(score)
