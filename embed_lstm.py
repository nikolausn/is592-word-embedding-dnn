#using tflearn make the graph creation simple
import tensorflow as tf
import tflearn
import numpy as np
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.model_selection import train_test_split

import pickle

with open('traintest.pickle', 'rb') as f:
    trainX,testX,trainY,testY = pickle.load(f)

with open('traintest-smote.pickle','rb') as f:
	train_smo_x,train_smo_y = pickle.load(f)

train_smo_y = to_categorical(train_smo_y,nb_classes=len(np.unique(train_smo_y)))

# Building the graph network
batch_size=1000
net = tflearn.input_data([None, 50])
# reduce dimensionality using word embedding transform the name structure into 1000 dimensions
net = tflearn.embedding(net,input_dim=62696,output_dim = 1000)
#net = tflearn.lstm(net,512, dropout=0.8, dynamic=True)
net = tflearn.lstm(net,1000, dropout=0.8)
net = tflearn.fully_connected(net, 23, activation='softmax')
net = tflearn.regression(net,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy')

# Training
#tf.reset_default_graph()
#with tf.Session() as session:
#	new_saver = tf.train.import_meta_graph('check/check_-.meta',clear_devices=True)
#	new_saver.restore(session, 'check/check_-360')
model = tflearn.DNN(net,tensorboard_verbose=0,best_checkpoint_path='./best_check2/best_',checkpoint_path='./check2/check_')
#with model.graph.as_default():
#	with tf.Session() as session:
#		new_saver = tf.train.import_meta_graph('check/check_-360.meta',clear_devices=True)
#		new_saver.restore(session, 'check/check_-360')

	#model.fit(train_smo_x,train_smo_y,epoch=30,validation_set=(testX,testY),show_metric=True,batch_size=batch_size,snapshot_epoch=True)
model.load('./check2/check_-792')
model.fit(trainX,trainY,n_epoch=30,validation_set=(testX,testY),show_metric=True,batch_size=batch_size,snapshot_epoch=True)

#model.save('./ethnicity.tflearn')
