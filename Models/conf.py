import os
import numpy as np
import pandas
from scipy.misc import imread, imresize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
import random
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, array_to_img

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DATA_PATH = '/home/zhaoyu/Dropbox/AI-II-8750/Project/Data/'
PROJECT_PATH = '/home/zhaoyu/Dropbox/AI-II-8750/Project/Code/deepLeaf/'

# DATA_PATH = '/home/zhaoyu/deepLeaf/Data/'
# PROJECT_PATH = '/media/zhaoyu/DATA/Dropbox/AI-II-8750/Project/Code/deepLeaf/'
