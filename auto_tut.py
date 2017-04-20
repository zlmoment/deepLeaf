from keras.layers import Input, Dense
from keras.models import Model

from scipy.misc import imread, imresize

from keras import regularizers

encoding_dim =32;

input_img = Input(shape =(784,))
#input_img = Input(shape =(1585,))

#encoded = Dense(encoding_dim, activation='relu',
#activity_regularizer=regularizers.l1(10e-5))(input_img)


#multilayer
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)


decoded = Dense(64, activation ='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
#decoded = Dense(1585, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img,encoded)

encoded_input = Input(shape=(encoding_dim,))

decode_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decode_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
import pandas
import numpy as np
import os, os.path

(x_train, _), (x_test, _) = mnist.load_data()

image_data=imresize(imread('./images/1.jpg'),(28,28)).astype(np.float32)
image_data=np.expand_dims(image_data,axis=0)
for img in os.listdir('./images'):
	resized_img = imresize(imread('./images/'+img),(28,28)).astype(np.float32)
	resized_img = np.expand_dims(resized_img, axis=0)
	image_data = np.vstack((image_data, resized_img))

print(image_data.shape)
print('shape')
print(x_train.shape)
x_train = image_data
x_test = image_data
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)



autoencoder.fit(x_train, x_train, epochs=100, batch_size=256, shuffle=True,
validation_data=(x_test, x_test))
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

import matplotlib.pyplot as plt

n=10
plt.figure(figsize=(20,4))
for i in range(n):
	ax = plt.subplot(2,n,i+1)
	plt.imshow(x_test[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(2,n, i+1+n)
	plt.imshow(decoded_imgs[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()




