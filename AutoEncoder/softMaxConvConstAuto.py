from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model
from scipy.misc import imread, imresize
from keras import backend as K

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(x)


x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)


decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

classified = Flatten()(encoded);
classified = Dense(99, activation='softmax')(classified) 

#autoencoder = Model(input_img, decoded)
autoencoder = Model(input_img, classified)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')




from keras.datasets import mnist
from keras.callbacks import TensorBoard
import numpy as np
import os, os.path

#(x_train, _), (x_test, _) = mnist.load_data()


image_data=imresize(imread('./../images/1.jpg'),(28,28)).astype(np.float32)
image_data=np.expand_dims(image_data,axis=0)
for img in os.listdir('./../images'):
        resized_img = imresize(imread('./../images/'+img),(28,28)).astype(np.float32)
        resized_img = np.expand_dims(resized_img, axis=0)
        image_data = np.vstack((image_data, resized_img))

#images 1585

x_train = image_data[1:1425]
x_test = image_data[1425:1585]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

print(x_train.shape)
print(x_test.shape)



autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


# encode and decode some digits
# note that we take them from the *test* set
#encoded_imgs = encoder.predict(x_test)
#decoded_imgs = decoder.predict(encoded_imgs)

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

decoded_imgs=autoencoder.predict(x_test)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i)
    plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
