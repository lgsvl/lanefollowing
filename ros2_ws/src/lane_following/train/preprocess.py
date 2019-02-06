import os
import cv2
import csv
import time
import h5py
import numpy as np
from utils import preprocess_image, CSV_PATH, IMG_PATH, HDF5_PATH


BIAS = 0.025
BATCH_SIZE = 10000


def split_data(train_test_ratio=0.8):
    images = []
    labels = []
    with open(os.path.join(CSV_PATH, 'training_data.csv')) as csvfile:
        lines = csvfile.readlines()
        reader = csv.reader(lines, delimiter=',')
        for line in reader:
            images.append(line[0])
            labels.append(line[1])

    images = np.array(images).reshape(-1, 1)
    labels = np.array(labels).reshape(-1, 1)

    data_size = images.shape[0]
    print('Total data size:', data_size)

    indices = np.random.permutation(data_size)
    train_size = int(round(data_size * train_test_ratio))
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    X_train = images[train_idx, :]
    Y_train = labels[train_idx, :]
    X_test = images[test_idx, :]
    Y_test = labels[test_idx, :]

    with open(os.path.join(CSV_PATH, 'train.txt'), 'w+') as f:
        for i in range(len(X_train)):
            f.write('{} {}\n'.format(X_train[i][0], Y_train[i][0]))

    with open(os.path.join(CSV_PATH, 'test.txt'), 'w+') as f:
        for i in range(len(X_test)):
            f.write('{} {}\n'.format(X_test[i][0], Y_test[i][0]))
    
    print('X_train:', X_train.shape)
    print('Y_train:', Y_train.shape)
    print('X_test:', X_test.shape)
    print('Y_test:', Y_test.shape)


def write_to_hdf5(phase='train'):
    with open (os.path.join(CSV_PATH, '{}.txt'.format(phase)), 'r') as f:
        data = f.readlines()[:]
    
    data_size = len(data) * 3
    i = 0

    print('{} data size:'.format(phase), data_size)

    images = []
    labels = []
    batch_idx = 0
    for row in data:
        for camera in ['center', 'left', 'right']:
            if camera == 'left':
                bias = BIAS
            elif camera == 'right':
                bias = -BIAS
            else:
                bias = 0.0
        
            img_id, label = row.split()

            img_path = os.path.join(IMG_PATH, '{}-{}.jpg'.format(camera, img_id))
            image = cv2.imread(img_path)
            image = preprocess_image(image, crop=True)
            label = float(label) + bias

            images.append(image)
            labels.append(label)

            i += 1
            if i % BATCH_SIZE == 0:
                h5_file = os.path.join(HDF5_PATH, '{}_{}.h5'.format(phase, batch_idx))
                X_data = np.array(images)
                Y_data = np.array(labels).reshape(-1, 1)
                with h5py.File(h5_file, 'w') as f:
                    f.create_dataset('data', data=X_data)
                    f.create_dataset('label', data=Y_data)
                
                with open(os.path.join(HDF5_PATH, '{}_h5_list.txt'.format(phase)), 'a') as f:
                    f.write(h5_file + '\n')
                
                images = []
                labels = []
                batch_idx += 1
                print(i)

    if i % BATCH_SIZE != 0:
        h5_file = os.path.join(HDF5_PATH, '{}_{}.h5'.format(phase, batch_idx))
        X_data = np.array(images)
        Y_data = np.array(labels).reshape(-1, 1)
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('data', data=X_data)
            f.create_dataset('label', data=Y_data)
        
        with open(os.path.join(HDF5_PATH, '{}_h5_list.txt'.format(phase)), 'a') as f:
            f.write(h5_file + '\n')
        images = []
        labels = []
        batch_idx += 1
        print(i)
    
    print('Done!')


if __name__ == '__main__':
    t0 = time.time()
    split_data(0.8)
    write_to_hdf5('train')
    write_to_hdf5('test')
    t1 = time.time()
    print('Elapsed time:', t1 - t0)
