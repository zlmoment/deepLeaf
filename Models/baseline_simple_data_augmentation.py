# in this model, images are padded first, then resize
# using cifar-10 model structure

from conf import *
import data_reader as data_reader

NB_EPOCHS = 200

train_csv, test_csv = data_reader.load_csv() 

# change data augmentation methods here
images = data_reader.load_image_data_padded_and_resize()
# load train labels (one-hot encoding)
train_labels_orig = data_reader.load_train_labels()

# separate train and test from images
train_images_orig = np.expand_dims(np.array([images[str(idx)] for idx in train_csv.id]), axis=4)
test_images = np.expand_dims(np.array([images[str(idx)] for idx in test_csv.id]), axis=4)
print(train_images_orig.shape)
print(test_images.shape)

# split original train into train and validation
nb_train = int(len(train_images_orig) * 0.9)
nb_val = len(train_images_orig) - nb_train
train_indices = random.sample(range(0, len(train_images_orig)), nb_train)
val_indices = [x for x in range(0, len(train_images_orig)) if x not in train_indices]

train_images = train_images_orig[train_indices, :, :, :]
train_labels = train_labels_orig[train_indices, :]
val_images = train_images_orig[val_indices, :, :, :]
val_labels = train_labels_orig[val_indices, :]

print('train_images shape', train_images.shape)
print('val_images shape', val_images.shape)

img_model = Sequential()

img_model.add(Conv2D(32, (3, 3), padding='same', input_shape=train_images.shape[1:]))
img_model.add(Activation('relu'))
img_model.add(Conv2D(32, (3, 3), padding='same'))
img_model.add(Activation('relu'))
img_model.add(MaxPooling2D(pool_size=(2, 2)))
img_model.add(Dropout(0.25))

img_model.add(Conv2D(64, (3, 3), padding='same'))
img_model.add(Activation('relu'))
img_model.add(Conv2D(64, (3, 3), padding='same'))
img_model.add(Activation('relu'))
img_model.add(MaxPooling2D(pool_size=(2, 2)))
img_model.add(Dropout(0.25))

img_model.add(Flatten())
img_model.add(Dense(512))
img_model.add(Activation('relu'))
img_model.add(Dropout(0.5))
img_model.add(Dense(99))
img_model.add(Activation('softmax'))

opt = keras.optimizers.adam(lr=0.0001, decay=1e-6)
img_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

img_model.count_params()
img_model.summary()

img_history = img_model.fit(train_images, train_labels,
                            batch_size=16,
                            epochs=NB_EPOCHS,
                            validation_data=(val_images, val_labels),
                            shuffle=True)

# plt.plot(img_history.history['acc'])
# plt.xlabel('Iterations')
# plt.ylabel('Training Accuracy')
# plt.title('Training Accuracy')
# plt.show()
#
# plt.plot(img_history.history['val_acc'])
# plt.xlabel('Iterations')
# plt.ylabel('Validation Accuracy')
# plt.title('Validation Accuracy')
# plt.show()

# plt.plot(img_history.history['acc'])
# plt.plot(img_history.history['val_acc'])
# plt.xlabel('Iterations')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
ax.plot(img_history.history['acc'])
ax.plot(img_history.history['val_acc'])
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(['train', 'validation'], loc='upper left')
fig.savefig('accuracy.png')   # save the figure to file
plt.close(fig)