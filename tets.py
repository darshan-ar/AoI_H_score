import numpy as np
from keras.layers import Input,BatchNormalization
from keras.utils import to_categorical
from sklearn import preprocessing
l = 5
A=20*l
fy1 = np.array([1,1,1,-1,-1,-1])
fy2 = np.array([-1,1,-1,1,-1,1])
#print(np.transpose(fy1).reshape(6,1)*fy2.reshape(1,6))
py1y2= (l*np.transpose(fy1).reshape(6,1)*fy2.reshape(1,6)) + A*np.ones((6,6)).astype(int)
#print(py1y2)
#print(np.shape(py1y2))
py1y2=preprocessing.normalize(py1y2,norm='l2')
py1y2.d
print(py1y2)
n = to_categorical(py1y2)
print(n)
