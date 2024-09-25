import tensorflow
!pip install tensorflow==2.15.0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import glob
from keras.utils import to_categorical
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten
import  tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_dir="/content/Dataset//train/"
test_dir="/content/Dataset/test/"
SIZE = 256

train_data_names = []
test_data_names = []

train_data = []
train_labels = []

for per in os.listdir(train_dir):
    for data in glob.glob(train_dir+per+'/*.*'):
        train_data_names.append(data)
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE,SIZE))
        train_data.append([img])
        if per[-1]=='g':
            train_labels.append(np.array(1))
        else:
            train_labels.append(np.array(0))

#Test Data

test_data = []
test_labels = []

for per in os.listdir(test_dir):
    for data in glob.glob(test_dir+per+'/*.*'):
        test_data_names.append(data)
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE,SIZE))
        test_data.append([img])
        if per[-1]=='g':
            test_labels.append(np.array(1))
        else:
            test_labels.append(np.array(0))

train_data = np.array(train_data)/255.0
train_labels = np.array(train_labels)
test_data = np.array(test_data)/255.0
test_labels = np.array(test_labels)

train_data = train_data.reshape(-1, SIZE,SIZE, 3)
test_data = test_data.reshape(-1, SIZE,SIZE, 3)
train_labels_c = to_categorical(train_labels)
test_labels_c = to_categorical(test_labels)

input_shape = (SIZE, SIZE, 3)
batch_size = 125
nr_epochs = 8
validation_split = 0.2
verbosity = 1
