import numpy as np
import tensorflow.compat.v1 as tf
from keras.utils import to_categorical
import tensorflow_datasets as tfds
from keras.layers import *
from keras import regularizers
from keras import Model
import keras.backend as K
import torch
epochs=20
l=1
A=1*l
nSamples = 6
n=20
fy1 = np.array([1,1,1,-1,-1,-1])
fy2 = np.array([-1,1,-1,1,-1,1])
p_y1y2= (l*np.transpose(fy1).reshape(6,1)*fy2.reshape(1,6)) + A*np.ones((6,6)).astype(int)
print(p_y1y2)


def GenerateY(P, U):
    # generate data according to the transition matrix P and the sequence U

    # retrun X sequence

    X = np.zeros((nSamples,), dtype=int)

    i = 0

    while (i < nSamples):
        #print("U,P", U[i], P[U[i],:])
        X[i] = np.random.choice(6, p=P[U[i], :])
        i = i + 1
    #print(X[:nSamples])
    yield X[:nSamples]

norm = np.reshape(np.sum(p_y1y2, axis=1), (6,1))
print("norm",norm)

Pxu = p_y1y2 / np.tile(norm, (1,6))  #each row is a valid distribution

print("Pxu",Pxu)
X = GenerateY(Pxu, fy1)
print("X",X)
Y=GenerateY(Pxu,fy2)
print("Y",Y)

#NN
x=tf.data.Dataset.from_generator(lambda: list(GenerateY(Pxu,fy1)),(tf.int64))
y=tf.data.Dataset.from_generator(lambda: list(GenerateY(Pxu,fy2)),(tf.int64))

y_1=[]
y_2=[]
print("Y1,Y2:",list(x.as_numpy_iterator()),list(y.as_numpy_iterator()))
print("Y1,Y2:",list(x.as_numpy_iterator()),list(y.as_numpy_iterator()))
for i in range(n):
    y_1.append(list(x.as_numpy_iterator())[0])
    y_2.append(list(y.as_numpy_iterator())[0])


print(np.shape(y_1),np.shape(y_2))

y1_train= to_categorical(y_1[:round(n-(n/10))])
print("Y1_onehot",y1_train)
y2_train = to_categorical(y_2[:round(n-(n/10))])
y1_valid=to_categorical(y_1[round(n-(n/10)):])
y2_valid=to_categorical(y_2[round(n-(n/10)):])
#tf_dataset=tf.data.Dataset.from_tensors((x,y))
#P= tf_dataset.batch(6)
#print("tf",list(P.as_numpy_iterator()))
# l=[]
# for o in range(100):
#     print(o)
#     print(list(P.as_numpy_iterator()))
# print("check: ",l)
#X=to_categorical(X)
#Y=to_categorical(Y)
#print(X,Y)


#NN



inputs = Input(shape=(6,6), name='one_hot')
x_0 = Dense(6, activation='relu',activity_regularizer=regularizers.l1(10e-5))(inputs)
x_0 = Flatten()(x_0)
x0 = Model(inputs,x_0)


w=x0.get_weights()

x_1 = Dense(6, activation='relu',activity_regularizer=regularizers.l1(10e-5))(inputs)
x_1 = Flatten()(x_1)
x1 = Model(inputs,x_1)

x1.set_weights(w)
def h_score(fx, gy):
    # compute the loss function -2*E[F(X)G(Y)] + tr(cov(F)*cov(G)
    #x0 = tf.Variable(0)
    x0 = x_0 - K.mean(x_0, 0)
    x0=K.cast(x0,'float32')

    x1 = x_1 - K.mean(x_1, 0)
    x1 = K.cast(x1, 'float32')
    #x1=tf.Variable(x1)
    Nsamples = K.size(x0[0])
    Nsamples = K.cast(Nsamples, 'float32')
    #Nsamples = tf.Variable(Nsamples)
    covf = K.dot(K.transpose(x0),x0) / Nsamples

    covg = K.dot(K.transpose(x1),x1) / Nsamples

    #h = -2 * K.sum(K.mean((x0 * x1)))
    #h= K.cast(K.sum(covf * covg,axis=1,),dtype='float32')

    h = -2 * K.mean(K.sum((x0 * x1),1)) + K.sum((covf * covg))
    return -h



c = 0
c1 = 0
for i in range(epochs):
    if i%2==0:
        c+=1
        print("Epoch: ",c,"from total epochs: ",i)
        x0 = Model(inputs, x_0)
        w = x0.get_weights()
        x0.compile(optimizer='adadelta', loss=h_score)
        fit_x0= x0.fit(y1_train,np.ones(np.shape(y1_train)),batch_size=250,epochs=1,shuffle=True)

    else:
        c1 += 1
        x1 = Model(inputs, x_1)
        x1.set_weights(w)
        print("Epoch: ", c1, "from total epochs: ", i)
        x1.compile(optimizer='adadelta', loss=h_score)
        fit_x1= x1.fit(y2_train,np.ones(np.shape(y2_train)),batch_size=250,epochs=1,shuffle=True)
        #print("X0,X1: ", x_0, x_1)

#covf = np.dot(np.cast(np.transpose(tf.make_tensor_proto(tf.make_ndarray(x_0.op.get_attr('value')))), 'int32'), np.cast(tf.make_tensor_proto(tf.make_ndarray(x_0.op.get_attr('value'))), 'int32')) / 6

#covg = np.dot(np.cast(np.transpose(tf.make_tensor_proto(tf.make_ndarray(x_1.op.get_attr('value')))), 'int32'), np.cast(tf.make_tensor_proto(tf.make_ndarray(x_1.op.get_attr('value'))), 'int32')) / 6
print("x0: ",x0.predict(y1_valid[0].reshape(30,6,6)),"x1: ",x1.predict(y2_valid[0].reshape(1,6,6)))