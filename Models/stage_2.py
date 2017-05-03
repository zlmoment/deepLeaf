from conf import *
import data_reader as data_reader

NB_EPOCHS = 1000

train_csv, test_csv = data_reader.load_csv()

# load train labels (one-hot encoding)
train_labels_orig = data_reader.load_train_labels()

train_feat, test_feat = data_reader.load_pre_extracted_features(standardize=True)

print('train_feat shape', train_feat.shape)
print('test_feat shape', test_feat.shape)
print('train_labels_orig shape', train_labels_orig.shape)

# train 10 times and get the average
val_acc_list = []
model_list = []
for i in range(10):
    feat_model = Sequential()
    feat_model.add(Dense(512, input_dim=192, kernel_initializer='uniform', activation='relu'))
    feat_model.add(Dropout(0.3))
    feat_model.add(Dense(256, activation='sigmoid'))
    feat_model.add(Dropout(0.3))
    feat_model.add(Dense(99, activation='softmax'))

    opt = keras.optimizers.adam(lr=0.0001, decay=1e-6)
    feat_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    feat_model.count_params()
    feat_model.summary()

    callbacks = [EarlyStopping(monitor='val_acc', patience=20, mode='max'),]
    feat_history = feat_model.fit(train_feat, train_labels_orig,
                                  batch_size=16,
                                  epochs=NB_EPOCHS,
                                  validation_split=0.1,
                                  shuffle=True,
                                  callbacks=callbacks)
    val_acc_list.append(feat_history.history['val_acc'][-1])
    model_list.append(feat_model)

for acc in val_acc_list:
    print(acc)

# prediction
best_model = model_list[val_acc_list.index(max(val_acc_list))]
pred = best_model.predict_proba(test_feat)
columns = sorted(train_csv.species.unique())
pred = pandas.DataFrame(pred, index=test_csv.id, columns=columns)
output = open('stage2.csv', 'w')
output.write(pred.to_csv())
output.close()

