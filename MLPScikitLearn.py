# -*- coding: utf-8 -*-
"""
Created on Thu May 17 18:29:20 2018

@author: sherwin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 16 22:28:23 2018

@author: sherwin
"""


from sklearn.neural_network import MLPClassifier

import scipy.io as sio

from sklearn.utils import shuffle
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import sklearn.preprocessing
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.externals import joblib
# Fit only to the training data

import numpy as np         # dealing with arrays
filename = 'D:/mlp/finalized_model1.sav'





label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(46))

train_x = sio.loadmat('D:\mlp\mlp_codebook.mat')
train_x=train_x['codebook']
train_y = sio.loadmat('D:\mlp\mlp_label.mat')
train_y=train_y['label']
train_y=np.transpose(train_y)
train_y=train_y-1

train_y = label_binarizer.transform(train_y)


train_x = np.reshape(train_x, (train_x.shape[0],64))
train_x = train_x.astype(np.float32)

train_y = np.reshape(train_y,(train_y.shape[0],46))

scaler.fit(train_x)
train_x = scaler.transform(train_x)


mlp = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(30,30,30), random_state=1,activation='logistic',learning_rate_init=0.001,max_iter=10000,tol=0)



mlp.fit(train_x,train_y)

joblib.dump(mlp, filename)





mlp = joblib.load(filename)

predictions = mlp.predict(train_x)

predictions =label_binarizer.inverse_transform(predictions)
print("new")
predictions=predictions+1