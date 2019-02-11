import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, Lambda, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from utils import load_multi_dataset, mkdir_p, HDF5_PATH, MODEL_PATH
from datetime import datetime
import time
from sklearn.model_selection import train_test_split


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

print('Loading data from HDF5...')
X_data, Y_data = load_multi_dataset(os.path.join(HDF5_PATH, 'train_h5_list.txt'))
# X_test, Y_test = load_multi_dataset(os.path.join(HDF5_PATH, 'test_h5_list.txt'))

print('Number of images:', X_data.shape[0])
print('Number of labels:', Y_data.shape[0])

print('Splitting data into training set and testing set....')
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0, input_shape=(70, 320, 3)))
model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()
model.compile(optimizer=Adam(lr=1e-04, decay=0.0), loss='mse')

t0 = time.time()
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), shuffle=True, epochs=30, batch_size=128)
t1 = time.time()
print('Total training time:', t1 - t0, 'seconds')

mkdir_p(MODEL_PATH)
model_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
model_file = os.path.join(MODEL_PATH, '{}.h5'.format(model_id))
model.save(model_file)
print("Training done successfully and model has been saved: {}".format(model_file))
print("Drive safely!")
