from conf import *

IMAGE_RESIZE_SIZE = 64


# read the train and test data

def load_csv():
    print('Loading CSV data...')
    train_csv = pandas.read_csv(DATA_PATH + '/train.csv')  # (990, 194)
    test_csv = pandas.read_csv(DATA_PATH + '/test.csv')  # (594, 193)

    return train_csv, test_csv


def load_train_labels():
    train_csv, test_csv = load_csv()
    le = LabelEncoder()
    le.fit(train_csv.species)
    train_labels_raw = le.transform(train_csv.species)
    # Convert class vectors to binary class matrices (one-hot encoding)
    train_labels = keras.utils.to_categorical(train_labels_raw, 99)

    return train_labels


def load_pre_extracted_features(standardize=False):
    train_csv, test_csv = load_csv()

    # get the pre-extracted features
    train_feat = train_csv.copy()
    test_feat = test_csv.copy()
    train_feat = train_feat.drop(['id', 'species'], axis=1)
    test_feat = test_feat.drop(['id'], axis=1)

    if standardize is True:
        # Standardize features by removing the mean and scaling to unit variance
        train_feat = StandardScaler().fit(train_feat).transform(train_feat)  # (990, 192)
        test_feat = StandardScaler().fit(test_feat).transform(test_feat)  # (594, 192)

    return train_feat, test_feat


def load_image_data_resize_directly():
    print('Loading images data...')
    image_data = {}
    for img_file in os.listdir(DATA_PATH + '/images'):
        resized_img = imresize(imread(DATA_PATH + '/images/' + img_file),
                               (IMAGE_RESIZE_SIZE, IMAGE_RESIZE_SIZE)).astype(np.float32)
        image_data[img_file.split(".")[0]] = resized_img

    return image_data


def load_image_data_padded_and_resize():
    print('Loading images data...')
    image_data = {}
    for img_file in os.listdir(DATA_PATH + '/images'):
        img = imread(DATA_PATH + '/images/' + img_file)
        h, w = img.shape
        max_dim = max(h, w)
        padded_img = np.lib.pad(img,
                                (((max_dim - h) // 2, max_dim - h - (max_dim - h) // 2),
                                 ((max_dim - w) // 2, max_dim - w - (max_dim - w) // 2)),
                                'constant', constant_values=1)
        resized_img = imresize(padded_img, (IMAGE_RESIZE_SIZE, IMAGE_RESIZE_SIZE)).astype(np.float32)
        image_data[img_file.split(".")[0]] = resized_img

    return image_data
