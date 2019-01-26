# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 14:36:12 2018

@author: sherwin
"""
import time
from datetime import timedelta
import math
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from timeit import default_timer as timer
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from tqdm import tqdm
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.utils import to_categorical

start = timer()

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


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(32, 32), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])


        ax.set_xlabel(xlabel)


        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


images = train_x[0:9]
class_true = np.argmax(train_y, axis=1)
class_true_test = np.argmax(test_y, axis=1)

cls_true = class_true[0:9]

plot_images(images=images, cls_true=cls_true)

img_size = 32

img_size_flat = img_size * img_size

img_shape = (img_size, img_size)

num_channels = 1

num_classes = 45


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,  # The previous layer.
                   num_input_channels,  # Num. channels in prev. layer.
                   filter_size,  # Width and height of each filter.
                   num_filters,  # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    if use_pooling:

        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)

    return layer, weights


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def new_fc_layer(input,  # The previous layer.
                 num_inputs,  # Num. inputs from prev. layer.
                 num_outputs,  # Num. outputs.
                 use_relu=True):  # Use Rectified Linear Unit (ReLU)?


    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


x_image = tf.placeholder('float', [None, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
                                            num_input_channels=num_channels,
                                            filter_size=filter_size1,
                                            num_filters=num_filters1,
                                            use_pooling=True)

layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                            num_input_channels=num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv2)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()

session.run(tf.global_variables_initializer())

train_batch_size = 64

epochs = 60


def optimize(hm_epochs):

    global total_iterations
    start_time = time.time()
    print("time Started at ", start_time)
    saver = tf.train.Saver()
    # saver.restore(sess, 'D:/Neural Network Model/trained.ckpt')
    for e in range(hm_epochs):

        i = 0
        while i < len(train_x):
            start = i
            end = i + train_batch_size
            x_batch = np.array(train_x[start:end])
            y_true_batch = np.array(train_y[start:end])
            feed_dict_train = {x_image: x_batch,
                               y_true: y_true_batch}

            session.run(optimizer, feed_dict=feed_dict_train)
            i += train_batch_size

        acc = session.run(accuracy, feed_dict=feed_dict_train)

        msg = "Epoch Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

        # Print it.
        print(msg.format(e, acc))


    save_path = saver.save(session, 'D:/Neural Network Model/trained.ckpt')
    print("Model saved in file: {}".format(save_path))
    end_time = time.time()

    time_dif = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def plot_example_errors(cls_pred, correct):

    incorrect = (correct == False)

    images = test_x[incorrect]


    cls_pred = cls_pred[incorrect]

    cls_true = class_true_test[incorrect]


    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def plot_confusion_matrix(cls_pred):

    cls_true = class_true_test

    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)

    plt.matshow(cm)

    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()


test_batch_size = 128


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    num_test = len(test_x)

    np.zeros(shape=num_test, dtype=np.int)


    i = 0

    while i < num_test:

        j = min(i + test_batch_size, num_test)

        images = test_x[i:j, :]

        # Get the associated labels.
        labels = test_y[i:j, :]

        feed_dict = {x_image: images,
                     y_true: labels}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    cls_true = class_true_test


    correct = (cls_true == cls_pred)

    correct_sum = correct.sum()

    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)


    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


optimize(epochs)
print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)


def singletest():
    training_data = []
    labels_train = []
    i = 0
    for img in tqdm(os.listdir(SingleTestDir)):
        label = label_img(img)
        path = os.path.join(SingleTestDir, img)
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
        i = i + 1
    value = np.array(labels_train)
    encoded2 = to_categorical(value, 45)
    return training_data, encoded2, i


images, labels, i = singletest()
images = np.reshape(images, (i, 32, 32, 1))

images = images.astype(np.float32)
labels = np.reshape(labels, (i, 45))

feed_dict = {x_image: images,
             y_true: labels}
cls_pred = []
cls_pred = session.run(y_pred_cls, feed_dict=feed_dict)
print(cls_pred)

session.close