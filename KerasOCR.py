# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:04:35 2018

@author: sherwin
"""

from sklearn.utils import shuffle

import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from tqdm import tqdm
import tensorflow as tf
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers

filter_size1 = 5  # Convolution filters are 5 x 5 pixels.
num_filters1 = 16  # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5  # Convolution filters are 5 x 5 pixels.
num_filters2 = 36  # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128

TRAIN_DIR = 'D:/Project/Training_set/'
TEST_DIR = 'D:/Project/Testing_set/'
SingleTestDir = 'D:/Project/single/'


# Assign labels and convert them into one-hot arrays
def label_img(img):
    word_label = img.split(' ')[0]

    return word_label


# Preprocess the training data
def create_train_data():
    training_data = []
    labels_train = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        old_size = img.shape[:2]  # old_size is in (height, width) format
        ratio = float(32) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = 32 - new_size[1]
        delta_h = 32 - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [255, 255, 255]
        # change this back to local variable
        new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        thresh = 127
        im_bw = cv2.threshold(new_im, thresh, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("image", im_bw)
        training_data.append([np.array(im_bw)])
        labels_train.append([np.array(label)])

    value = np.array(labels_train)
    encoded1 = to_categorical(value, 45)
    return training_data, encoded1


def create_test_data():
    testing_data = []
    labels_test = []
    for img in tqdm(os.listdir(TEST_DIR)):
        label = label_img(img)
        path = os.path.join(TEST_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        old_size = img.shape[:2]  # old_size is in (height, width) format
        ratio = float(32) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = 32 - new_size[1]
        delta_h = 32 - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [255, 255, 255]
        new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        thresh = 127
        im_bw = cv2.threshold(new_im, thresh, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("image", im_bw)
        testing_data.append([np.array(im_bw)])
        labels_test.append([np.array(label)])

    value = np.array(labels_test)
    encoded2 = to_categorical(value, 45)
    return testing_data, encoded2


train_x, train_y = create_train_data()
test_x, test_y = create_test_data()
train_x = np.reshape(train_x, (1056, 32, 32, 1))

train_x = train_x.astype(np.float32)
train_y = np.reshape(train_y, (1056, 45))
test_x = np.reshape(test_x, (264, 32, 32, 1))
test_x = test_x.astype(np.float32)
test_y = np.reshape(test_y, (264, 45))

train_x, train_y = shuffle(train_x, train_y, random_state=0)
test_x, test_y = shuffle(test_x, test_y, random_state=0)

train_x = train_x / 255.0
test_x = test_x / 255.0

# Create the model
model = Sequential()
model.add(Conv2D(64, (5, 5), input_shape=(32, 32, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))

model.add(Conv2D(32, (5, 5), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))

model.add(Conv2D(16, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(8, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(45, activation='softmax'))
# Compile model
epochs = 1000
lrate = 0.9
decay = lrate / epochs

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=0.000001, decay=decay,
                                              amsgrad=False), metrics=['accuracy'])
print(model.summary())

model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=512)

# Final evaluation of the model
scores = model.evaluate(test_x, test_y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))