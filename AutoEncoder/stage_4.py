from conf import *
import data_reader as data_reader

NB_EPOCHS = 1000

train_csv, test_csv = data_reader.load_csv()

# change data augmentation methods here
# images = data_reader.load_image_data_resize_directly()
images = data_reader.load_image_data_padded_and_resize()
# load train labels (one-hot encoding)
train_labels_orig = data_reader.load_train_labels()

train_feat, test_feat = data_reader.load_pre_extracted_features(standardize=True)

# separate train and test from images
train_images_orig = np.expand_dims(np.array([images[str(idx)] for idx in train_csv.id]), axis=4)
test_images = np.expand_dims(np.array([images[str(idx)] for idx in test_csv.id]), axis=4)
print(train_images_orig.shape)
print(test_images.shape)

# train 10 times and get the average
val_acc_list = []
model_list = []
for i in range(10):
    # split original train into train and validation
    nb_train = int(len(train_images_orig) * 0.8)
    nb_val = len(train_images_orig) - nb_train
    train_indices = random.sample(range(0, len(train_images_orig)), nb_train)
    val_indices = [x for x in range(0, len(train_images_orig)) if x not in train_indices]

    train_images = train_images_orig[train_indices, :, :, :]
    train_feats = train_feat[train_indices, :]
    train_labels = train_labels_orig[train_indices, :]
    val_images = train_images_orig[val_indices, :, :, :]
    val_feats = train_feat[val_indices, :]
    val_labels = train_labels_orig[val_indices, :]

    print('train_images shape', train_images.shape)
    print('train_labels shape', train_labels.shape)
    print('train_feats shape', train_feats.shape)
    print('val_feats shape', val_feats.shape)


    
#######    #features extracted
    input_img = Input(shape=(64, 64, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    
    encoder = Model(input_img, encoded)

    
    autoencoder.fit(train_images, train_images,
                    epochs=50,
                    shuffle=True,
                    validation_data=(val_images, val_images))

    

    encoded_train_images = encoder.predict(train_images)


########
    encoded_input = Input(shape=(64,64,16,))
    flt           = Flatten()(encoded_input)
    encoded_feat  = Dense(256, activation='sigmoid')(flt)
#####3      



    image_inputs = Input(shape=train_images.shape[1:])
    x = Conv2D(4, (9, 9), padding='same')(image_inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Conv2D(8, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    feat_input = Input(shape=train_feats.shape[1:])
    x2 = Dense(512, input_dim=192, kernel_initializer='uniform', activation='relu')(feat_input)
    x2 = Dropout(0.3)(x2)
    x2 = Dense(256, activation='sigmoid')(x2)
    x2 = Dropout(0.3)(x2)
    


    print('x shape', x.shape)
    print('x2 shape', x2.shape)

    concat = keras.layers.concatenate([x, x2, encoded_feat], axis=1)
    concat = Dense(256)(concat)
    output = Dense(99, activation='softmax')(concat)

    complex_model = Model(inputs=[image_inputs, feat_input, input_img], outputs=output)

    opt = keras.optimizers.adam(lr=0.0001, decay=1e-6)
    complex_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    complex_model.count_params()
    complex_model.summary()

    callbacks = [EarlyStopping(monitor='val_acc', patience=20, mode='max'),]

    history = complex_model.fit([train_images, train_feats], train_labels,
                                batch_size=16,
                                epochs=NB_EPOCHS,
                                validation_data=([val_images, val_feats], val_labels),
                                shuffle=True,
                                callbacks=callbacks)
    val_acc_list.append(history.history['val_acc'][-1])
    model_list.append(complex_model)

for acc in val_acc_list:
    print(acc)

# prediction
best_model = model_list[val_acc_list.index(max(val_acc_list))]
pred = best_model.predict([test_images, test_feat])
columns = sorted(train_csv.species.unique())
pred = pandas.DataFrame(pred, index=test_csv.id, columns=columns)
output = open('stage3.csv','w')
output.write(pred.to_csv())
output.close()
