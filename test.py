
from keras.datasets import mnist
import pandas
import numpy as np
import os, os.path
from  scipy.misc import imread, imresize

(x_train, _), (x_test, _) = mnist.load_data()

image_data={}
for img in os.listdir('./images'):
        resized_img = imresize(imread('./images/'+img),(128,128)).astype(np.float32)
        image_data[img.split(".")[0]] = resized_img


print(type(image_data));
print(x_train.ndim);
print(x_train.size);
print(x_train.dtype);
print(type(x_train));





image_data=imresize(imread('./images/1.jpg'),(28,28)).astype(np.float32)
image_data=np.expand_dims(image_data,axis=0)
for img in os.listdir('./images'):
        resized_img = imresize(imread('./images/'+img),(28,28)).astype(np.float32)
        resized_img = np.expand_dims(resized_img, axis=0)
        image_data = np.vstack((image_data, resized_img))

x_train = image_data[1:1425];


print('image');
print(image_data.shape);
print(x_train.shape);



