import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns

tf.keras.backend.clear_session()
#Adding Seed so that random initialization is consistent
import os
from numpy.random import seed
seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)
import dataset
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model

#Prepare input data
train_path='training_data'
classes = ['video_frames']
print(classes)
num_classes = len(classes)

# 10% of the data will automatically be used for validation
validation_size = 0.1
img_size = 200
num_channels = 3
sample_size = 50

data = dataset.read_train_sets(train_path, img_size, ['video_frames'], validation_size=validation_size, sample_size=sample_size)


print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))
from tensorflow import keras
from tensorflow.keras import layers

total_pixels = img_size * img_size * 3
translator_factor = 2
translator_layer_size = 150
middle_factor = 2
middle_layer_size = 100

inputs = Input(shape=(img_size,img_size,3), name='cat_image')
x = Conv2D(64, (3, 3), padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(16, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(3, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
decoded = Activation('sigmoid')(x)

model = Model(inputs=inputs, outputs=decoded)
customAdam = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=customAdam,  # Optimizer
              # Loss function to minimize
              loss="mean_squared_error",
              # List of metrics to monitor
              metrics=["mean_squared_error"])

x_train, _, _, _ = data.train.next_batch(20)
x_valid, _, _, _ = data.valid.next_batch(10)
#print(x_valid)

print('# Fit model on training data')

history = model.fit(x_train,
                    x_train, #we pass it the input itself as desired output
                    batch_size=250,
                    epochs=10,
                    # We pass it validation data to
                    # monitor loss and metrics
                    # at the end of each epoch
                    validation_data=(x_valid, x_valid))


import matplotlib.pyplot as plt
plt.figure(figsize=(20, 4))
valid_predictions = model.predict(x_valid[:40])
print(valid_predictions)

instance = x_valid[0]
decoded_img = valid_predictions[0]
print(decoded_img)
ax = plt.subplot(1, 2, 1)
plt.imshow(instance)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

subplot = plt.subplot(1, 2, 2)
plt.imshow(decoded_img)

subplot.get_xaxis().set_visible(False)
subplot.get_yaxis().set_visible(False)
plt.show()