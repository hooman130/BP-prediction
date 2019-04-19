import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Masking
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
import keras
import os
import tensorflow as tf
import keras.backend as K

#length of the padded sequences
series_len = 200

#importing the labels
f = open('Subject3/labels.pckl', 'rb')
labels = pickle.load(f, encoding='latin1')
f.close()

#importing the data
f = open('Subject3/list_data.pckl', 'rb')
list_data = pickle.load(f, encoding='latin1')
f.close()

num_features = 4
#normalizing the labels
scaler = MinMaxScaler(feature_range=(0, 1))
labels = np.array(labels)
labels_scaled = scaler.fit_transform(labels)

np.random.seed(1)
#portioning data into bins and selecting training and test sets
bin_len = 2
bins = int(len(list_data)/bin_len)
idx_bins = np.sort(np.random.choice(bins,int(bins*0.8),replace=False))
idx2 = np.zeros(0,dtype=int)
for i in range(idx_bins.shape[0]):
    idx2 = np.concatenate((idx2, np.array(range(idx_bins[i]*bin_len, (idx_bins[i]+1)*bin_len ))))

#concatinating all train data
all_train2 = np.zeros((0,num_features))
for i in idx2:
    all_train2 = np.concatenate((all_train2, list_data[i]))
#fitting the scaler to the training data
standard_data_scaler2 = StandardScaler()
standard_data_scaler2.fit(all_train2)
#transformig and padding all data
data2 = np.zeros((0,series_len,num_features))
for i in range(len(list_data)):
    l = list_data[i].shape[0]
    data2 = np.concatenate((data2,np.pad(standard_data_scaler2.transform(list_data[i]), ((series_len-l,0),(0,0)), 'constant',
                                       constant_values = -3 ).reshape(1,series_len,num_features)), axis=0)

train_data2 = data2[idx2,:,:]
test_data2 = np.delete(data2,idx2,axis=0)
train_labels_sbp2 = labels_scaled[idx2,0]
train_labels_dbp2 = labels_scaled[idx2,1]
test_labels_sbp2 = np.delete(labels_scaled,idx2,axis=0)[:,0]
test_labels_dbp2 = np.delete(labels_scaled,idx2,axis=0)[:,1]
test_labels = np.delete(labels,idx2,axis=0)

# defining the R metric
def correlation_coefficient(y_true, y_pred):
    x = tf.cast(y_true, dtype=tf.float64)
    y = tf.cast(y_pred, dtype=tf.float64)
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my 
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return r

#building the model
inputs = Input(shape=(series_len, num_features))
mask = Masking(mask_value=-3)(inputs)#masking layer
lstm1 = LSTM(15, return_sequences= False, dropout=0.25, recurrent_dropout=0.0)(mask)#the shared layer
#sbp dense layers
dense_sbp = Dense(5, activation='relu')(lstm1)
dense_sbp_2 = Dense(5, activation='relu')(dense_sbp)
dense_sbp_3 = Dense(5, activation='relu')(dense_sbp_2)
dense_sbp_4 = Dense(5, activation='relu')(dense_sbp_3)
dense_sbp_5 = Dense(5, activation='relu')(dense_sbp_4)
#dbp dense layers
dense_dbp = Dense(5, activation='relu')(lstm1)
dense_dbp_2 = Dense(5, activation='relu')(dense_dbp)
dense_dbp_3 = Dense(5, activation='relu')(dense_dbp_2)
dense_dbp_4 = Dense(5, activation='relu')(dense_dbp_3)
dense_dbp_5 = Dense(5, activation='relu')(dense_dbp_4)

output_sbp = Dense(1, activation='linear')(dense_sbp_5)
output_dbp = Dense(1, activation='linear')(dense_dbp_5)
model =Model(input=inputs, output = [output_sbp,output_dbp])
adam = optimizers.Adam(lr=0.0024, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=[correlation_coefficient])
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=30, verbose=1, min_lr=1e-15)
early_stopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=150, verbose=1)
callbacks_list = [reduce_lr, early_stopping]
history = model.fit(train_data2, [train_labels_sbp2,train_labels_dbp2], epochs=2000,
                    batch_size=280, verbose=2, callbacks=callbacks_list,
                    validation_data =(test_data2, [test_labels_sbp2,test_labels_dbp2]))
model.save('model3.h5', overwrite=True)
#evaluating the model on the test set
pred_scaled = model.predict(test_data2)#predicting the test output
temp=np.concatenate((pred_scaled,pred_scaled), axis=1)
pred = scaler.inverse_transform(temp)#transforming the predicted labels back to the original range

mae_sbp = mean_absolute_error(test_labels[:,0],pred[:,0])
r_sbp = pearsonr(test_labels[:,0],pred[:,0])[0]
mae_dbp = mean_absolute_error(test_labels[:,1],pred[:,1])
r_dbp = pearsonr(test_labels[:,1],pred[:,1])[0]