import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from utils import load_multi_dataset, mkdir_p, HDF5_PATH, MODEL_PATH
from datetime import datetime
import time


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

X_train, Y_train = load_multi_dataset(os.path.join(HDF5_PATH, 'train_h5_list.txt'))
X_test, Y_test = load_multi_dataset(os.path.join(HDF5_PATH, 'test_h5_list.txt'))

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
model.compile(optimizer=Adam(lr=0.0001, decay=0.0), loss='mse')

t0 = time.time()
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), shuffle=True, epochs=30, batch_size=256)
t1 = time.time()
print('Total training time:', t1 - t0, 'seconds')

mkdir_p(MODEL_PATH)
model_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
model_file = os.path.join(MODEL_PATH, '{}.h5'.format(model_id))
model.save(model_file)
print("Saved model: {}".format(model_file))
print("Drive safely!")
