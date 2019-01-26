# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:29:52 2018

@author: sherwin
"""

# Import libraries and modules
from timeit import default_timer as timer
# working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import scipy.io as sio
from keras.utils import to_categorical
from sklearn.utils import shuffle

start = timer()

train_x = sio.loadmat('D:\Project\Codebook.mat')
train_x = train_x['codebook']
train_y = sio.loadmat('D:\Project\Labels.mat')
train_y = train_y['label']
test_x = sio.loadmat('D:\Project\Codebook_test.mat')
test_x = test_x['codebook2']
test_y = sio.loadmat('D:\Project\Labels_test.mat')
test_y = test_y['label2']

train_y = to_categorical(train_y, 45)
test_y = to_categorical(test_y, 45)

train_x = np.reshape(train_x, (1056, 159))

train_x = train_x.astype(np.float32)
train_y = np.reshape(train_y, (1056, 45))
test_x = np.reshape(test_x, (264, 159))
test_x = test_x.astype(np.float32)
test_y = np.reshape(test_y, (264, 45))

train_x, train_y = shuffle(train_x, train_y, random_state=0)
test_x, test_y = shuffle(test_x, test_y, random_state=0)

# We'll take 3 hidden layers
n_nodes_hl1 = 200  # Number of nodes in each hidden layer
n_nodes_hl2 = 250
n_nodes_hl3 = 100

n_classes = 45  # Number of output categories
batch_size = 512  # Number of training examples to go through in one iteration
hm_epochs = 20  # Number of times to run through the entire training data
lr = 0.1
epsilon = 1e-1

x = tf.placeholder('float', [None, 159])
y = tf.placeholder('float')

hidden_1_layer = {'f_fum': n_nodes_hl1,
                  'weight': tf.Variable(tf.random_normal([159, n_nodes_hl1])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}  # define structure of different layers

hidden_2_layer = {'f_fum': n_nodes_hl2,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum': n_nodes_hl3,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum': None,
                'weight': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes])), }

# Model of the neural network : 1 input layer -> 3 hidden layers -> 1 output layer


l1 = tf.add(tf.matmul(x, hidden_1_layer['weight']), hidden_1_layer['bias'])
batch_mean1, batch_var1 = tf.nn.moments(l1, [0])
scale1 = tf.Variable(tf.ones([n_nodes_hl1]))
beta1 = tf.Variable(tf.zeros([n_nodes_hl1]))
l1 = tf.nn.batch_normalization(l1, batch_mean1, batch_var1, beta1, scale1, epsilon)
l1 = tf.nn.sigmoid(l1)

l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
batch_mean2, batch_var2 = tf.nn.moments(l2, [0])
scale2 = tf.Variable(tf.ones([n_nodes_hl2]))
beta2 = tf.Variable(tf.zeros([n_nodes_hl2]))
l2 = tf.nn.batch_normalization(l2, batch_mean2, batch_var2, beta2, scale2, epsilon)
l2 = tf.nn.sigmoid(l2)

l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias'])
batch_mean3, batch_var3 = tf.nn.moments(l3, [0])
scale3 = tf.Variable(tf.ones([n_nodes_hl3]))
beta3 = tf.Variable(tf.zeros([n_nodes_hl3]))
l3 = tf.nn.batch_normalization(l3, batch_mean3, batch_var3, beta3, scale3, epsilon)
l3 = tf.nn.sigmoid(l3)

output = tf.matmul(l3, output_layer['weight']) + output_layer['bias']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(hm_epochs):
        avg_loss = 0
        i = 0
        while i < len(train_x):
            start = i
            end = i + batch_size
            batch_x = np.array(train_x[start:end])
            batch_y = np.array(train_y[start:end])

            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_loss += c / len(train_x)
            i += batch_size

        print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_loss))
        pred_temp = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
        print("Validation Accuracy:", accuracy.eval({x: test_x, y: test_y}))

        predict = tf.argmax(output, 1)
        pred = predict.eval({x: test_x})
    save_path = saver.save(sess, 'D:/NNFeatures/trained.ckpt')
    print("Model saved in file: {}".format(save_path))
    pred_temp = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    print("Validation Accuracy:", accuracy.eval({x: test_x, y: test_y}))

    predict = tf.argmax(output, 1)
    pred = predict.eval({x: test_x})

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, 'D:/NNFeatures/trained.ckpt')
    pred_temp = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    print("Validation Accuracy:", accuracy.eval({x: test_x, y: test_y}, session=sess))

    predict = tf.argmax(output, 1)
    pred = predict.eval({x: test_x})# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:29:52 2018

@author: sherwin
"""



# Import libraries and modules
from timeit import default_timer as timer
               # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import scipy.io as sio
from keras.utils import to_categorical
from sklearn.utils import shuffle

start = timer()

train_x = sio.loadmat('D:\Project\Codebook.mat')
train_x=train_x['codebook']
train_y = sio.loadmat('D:\Project\Labels.mat')
train_y=train_y['label']
test_x = sio.loadmat('D:\Project\Codebook_test.mat')
test_x=test_x['codebook2']
test_y= sio.loadmat('D:\Project\Labels_test.mat')
test_y=test_y['label2']





train_y = to_categorical(train_y,45)
test_y=to_categorical(test_y,45)




train_x = np.reshape(train_x, (1056,159))

train_x = train_x.astype(np.float32)
train_y = np.reshape(train_y, (1056, 45))
test_x = np.reshape(test_x, (264,159))
test_x = test_x.astype(np.float32)
test_y = np.reshape(test_y, (264,45))






train_x, train_y = shuffle(train_x, train_y, random_state=0)
test_x, test_y = shuffle(test_x, test_y, random_state=0)

# We'll take 3 hidden layers
n_nodes_hl1 = 200                            # Number of nodes in each hidden layer
n_nodes_hl2 = 250
n_nodes_hl3 = 100

n_classes = 45                                # Number of output categories
batch_size = 512                              # Number of training examples to go through in one iteration
hm_epochs = 20                         # Number of times to run through the entire training data
lr = 0.1
epsilon = 1e-1

x = tf.placeholder('float', [None, 159])
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([159, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}     # define structure of different layers

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                 'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                 'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}


# Model of the neural network : 1 input layer -> 3 hidden layers -> 1 output layer


l1 = tf.add(tf.matmul(x,hidden_1_layer['weight']), hidden_1_layer['bias'])
batch_mean1, batch_var1 = tf.nn.moments(l1,[0])
scale1 = tf.Variable(tf.ones([n_nodes_hl1]))
beta1 = tf.Variable(tf.zeros([n_nodes_hl1]))
l1 = tf.nn.batch_normalization(l1,batch_mean1,batch_var1,beta1,scale1,epsilon)
l1 = tf.nn.sigmoid(l1)

l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
batch_mean2, batch_var2 = tf.nn.moments(l2,[0])
scale2 = tf.Variable(tf.ones([n_nodes_hl2]))
beta2 = tf.Variable(tf.zeros([n_nodes_hl2]))
l2 = tf.nn.batch_normalization(l2,batch_mean2,batch_var2,beta2,scale2,epsilon)
l2 = tf.nn.sigmoid(l2)

l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
batch_mean3, batch_var3 = tf.nn.moments(l3,[0])
scale3 = tf.Variable(tf.ones([n_nodes_hl3]))
beta3 = tf.Variable(tf.zeros([n_nodes_hl3]))
l3= tf.nn.batch_normalization(l3,batch_mean3,batch_var3,beta3,scale3,epsilon)
l3 = tf.nn.sigmoid(l3)

output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels= y))

optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
        sess.run(init)
        for epoch in range(hm_epochs):
            avg_loss = 0
            i=0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                avg_loss += c/len(train_x)
                i+=batch_size

            print( "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_loss))
            pred_temp = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
            print ("Validation Accuracy:",accuracy.eval({x:test_x, y:test_y}))

            predict = tf.argmax(output, 1)
            pred = predict.eval({x: test_x})
        save_path = saver.save(sess, 'D:/NNFeatures/trained.ckpt')
        print("Model saved in file: {}".format(save_path))
        pred_temp = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
        print ("Validation Accuracy:",accuracy.eval({x:test_x, y:test_y}))

        predict = tf.argmax(output, 1)
        pred = predict.eval({x: test_x})




with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, 'D:/NNFeatures/trained.ckpt')
        pred_temp = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
        print ("Validation Accuracy:",accuracy.eval({x:test_x, y:test_y},session=sess))

        predict = tf.argmax(output, 1)
        pred = predict.eval({x: test_x})