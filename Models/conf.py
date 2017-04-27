import os
import numpy as np
import pandas
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import random
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, array_to_img

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DATA_PATH = '/home/zhaoyu/Dropbox/AI-II-8750/Project/Data/'
PROJECT_PATH = '/home/zhaoyu/Dropbox/AI-II-8750/Project/Code/deepLeaf/'
