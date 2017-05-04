from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model
from keras.callbacks import TensorBoard
from scipy.misc import imread, imresize
import numpy as np
import os, os.path
import matplotlib.pyplot as plt
import data_reader as data_reader
import pydot
from keras.utils import plot_model
import random


#load data
#loading images
image_data=imresize(imread('./../images/1.jpg'),(64,64)).astype(np.float32)
image_data=np.expand_dims(image_data,axis=0)
for img in os.listdir('./../images'):
        resized_img = imresize(imread('./../images/'+img),(64,64)).astype(np.float32)
        resized_img = np.expand_dims(resized_img, axis=0)
        image_data = np.vstack((image_data, resized_img))



#loading labels
train_csv, test_csv = data_reader.load_csv()
print(train_csv.shape);
print(test_csv.shape);

images = data_reader.load_image_data_padded_and_resize()
train_images_orig = np.expand_dims(np.array([images[str(idx)] for idx in train_csv.id]), axis=4)
train_labels_orig = data_reader.load_train_labels()



val_acc_list = []
model_list = []


for i in range(10):
    #for validation
    nb_train = int(len(train_images_orig) * 0.8)
    nb_val = len(train_images_orig) - nb_train
    train_indices = random.sample(range(0, len(train_images_orig)), nb_train)
    val_indices = [x for x in range(0, len(train_images_orig)) if x not in train_indices]

    train_images = train_images_orig[train_indices, :, :, :]
    train_labels = train_labels_orig[train_indices, :]
    val_images = train_images_orig[val_indices, :, :, :]
    val_labels = train_labels_orig[val_indices, :]



#Build autoencoder

    input_img = Input(shape=(64,64,1))
    encoded  = Conv2D(16,(3,3),activation='relu', padding='same')(input_img)
    decoded  = Conv2D(1,(3,3),activation='relu', padding='same')(encoded)

    autoencoder = Model(input_img, decoded) 
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

#internal layer
    encoder = Model(input_img, encoded)



  #visual

    plot_model(autoencoder, to_file='model.png')
  #train autoencoder

    img_hist = autoencoder.fit(train_images, train_images,#x_train, x_train,
		epochs=50,
		batch_size=128,
		shuffle=True,
		validation_data=(val_images, val_images),
		callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]
		)




    decoded_imgs = autoencoder.predict(val_images)
    val_acc_list.append(img_hist.history['val_acc'][-1])
	
for acc in val_acc_list:
   print(acc)
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(val_images[i].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



#encoded stuff
encoded_train_images = encoder.predict(train_images)
encoded_val_images  = encoder.predict(val_images)
print(encoded_val_images.shape)


#softmax categorizer
encoded_input = Input(shape=(64,64,16,))
flt           = Flatten()(encoded_input)
middle        = Dense(256, activation='sigmoid') (flt)
classifier    = Dense(99, activation='softmax')(middle)

clasModel    = Model(encoded_input, classifier)

clasModel.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
clasModel.fit(encoded_train_images, train_labels,
		epochs=50,
		batch_size=128,
		shuffle=True,
		validation_data=(encoded_val_images, val_labels),
		)



print('all good sofar')
#build classification training










