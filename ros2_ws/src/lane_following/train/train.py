import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from utils import load_multi_dataset, HDF5_PATH


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

X_train, Y_train = load_multi_dataset(os.path.join(HDF5_PATH, 'train_h5_list.txt'))
X_test, Y_test = load_multi_dataset(os.path.join(HDF5_PATH, '/test_h5_list.txt'))

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

model = Sequential()
model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation='relu', input_shape=(66, 200, 3)))
model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(lr=0.0001, decay=0.0), loss='mse', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), shuffle=True, epochs=3, batch_size=256)
model.save('model/model.h5')
