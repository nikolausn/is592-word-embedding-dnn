# using tflearn make the graph creation simple
import tensorflow as tf
import tflearn
import numpy as np
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.model_selection import train_test_split

import pickle

with open('traintest.pickle', 'rb') as f:
    trainX,testX,trainY,testY = pickle.load(f)

with open('traintest-smote.pickle','rb') as f:
	train_res,test_res = pickle.load(f)

test_res = to_categorical(test_res,nb_classes=len(np.unique(test_res)))

# Building the graph network
batch_size=100
net = tflearn.input_data([None, 50])
# reduce dimensionality using word embedding transform the name structure into 1000 dimensions
net = tflearn.embedding(net,input_dim=62696,output_dim = 1000)
#net = tflearn.lstm(net,512, dropout=0.8, dynamic=True)
net = tflearn.lstm(net,1000, dropout=0.8)
net = tflearn.fully_connected(net, 23, activation='softmax')
net = tflearn.regression(net,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net,tensorboard_verbose=0,best_checkpoint_path='./best_check/best_',checkpoint_path='./check/check_')

model.fit(trainX,trainY,validation_set=(testX,testY),show_metric=True,batch_size=batch_size,snapshot_epoch=True)
