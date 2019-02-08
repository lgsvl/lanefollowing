# ROS2 End-to-End Lane Following Model with LGSVL Simulator

This documentation describes applying a deep learning neural network for lane following in [LGSVL Simulator](https://www.lgsvlsimulator.com/). In this project, we use LGSVL Simulator for customizing sensors (one main camera and two side cameras) for a car, collect data for training, and deploying and testing a trained model.

> This project was inspired by [NVIDIA's End-to-End Deep Learning Model for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

## Table of Contents

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
	- [Installing Docker CE](#installing-docker-ce)
	- [Installing NVIDIA Docker](#installing-nvidia-docker)
	- [Pulling Docker Image](#pulling-docker-image)
	- [What's inside Docker Image](#whats-inside-docker-image)
- [Features](#features)
- [Training Details](#training-details)
	- [Network Architecture](#network-architecture)
	- [Hyperparameters](#hyperparameters)
	- [Dataset](#dataset)
- [How to Collect Data and Train Your Own Model with LGSVL Simulator](#how-to-collect-data-and-train-your-own-model-with-lgsvl-simulator)
	- [Collect data from LGSVL Simulator](#collect-data-from-lgsvl-simulator)
	- [Data preprocessing](#data-preprocessing)
	- [Train a model](#train-a-model)
	- [Drive with your trained model in LGSVL Simulator](#drive-with-your-trained-model-in-lgsvl-simulator)
- [Future Works and Contributing](#future-works-and-contributing)
- [Links](#links)

## Getting Started

First, clone the repository:

```
git clone --recurse-submodules https://github.com/lgsvl/lanefollowing.git
```

Next, pull the latest Docker image and add a tag:

```
docker pull {docker image url}
docker tag {docker image url} lgsvl/lanefollowing:latest
```

Dataset
To build ROS2 packages:

```
docker-compose up build
```

Now, launch the lane following model:

```
docker-compose up drive
```

That's it! Now, the lane following ROS2 node and the rosbridge should be up and running, waiting for LGSVL Simulator to connect.

And, this is how driving looks like on San Francisco bridge in LGSVL Simulator.
[VIDEO goes here]

## Prerequisites

- Docker CE
- NVIDIA Docker
- NVIDIA graphics card (required for training/inference with GPU)

## Setup

### Installing Docker CE

To install Docker CE please refer to the [official documentation](https://docs.docker.com/install/linux/docker-ce/ubuntu/). We also suggest following through with the [post installation steps](https://docs.docker.com/install/linux/linux-postinstall/).

### Installing NVIDIA Docker

Before installing nvidia-docker make sure that you have an appropriate NVIDIA driver installed.
To test if NVIDIA drivers are properly installed enter `nvidia-smi` in a terminal. If the drivers are installed properly an output similar to the following should appear.
```
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 390.87                 Driver Version: 390.87                    |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GeForce GTX 108...  Off  | 00000000:65:00.0  On |                  N/A |
    |  0%   59C    P5    22W / 250W |   1490MiB / 11175MiB |      4%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0      1187      G   /usr/lib/xorg/Xorg                           863MiB |
    |    0      3816      G   /usr/bin/gnome-shell                         305MiB |
    |    0      4161      G   ...-token=7171B24E50C2F2C595566F55F1E4D257    68MiB |
    |    0      4480      G   ...quest-channel-token=3330599186510203656   147MiB |
    |    0     17936      G   ...-token=5299D28BAAD9F3087B25687A764851BB   103MiB |
    +-----------------------------------------------------------------------------+
```

The installation steps for nvidia-docker are available at the [official repo](https://github.com/NVIDIA/nvidia-docker).

### Pulling Docker Image

```
docker pull {docker image url}
docker tag {docker image url} lgsvl/lanefollowing:latest
```

### What's inside Docker Image

- Ubuntu 18.04
- CUDA 9.2
- cuDNN 7.1.4.18
- Python 3.6
- TensorFlow 1.8
- Keras 2.2.4
- ROS2 Crystal + rosbridge
- Jupyter Notebook

## Features

- Training mode: Manually drive the vehicle and collect data
- Autonomous Mode: The vehicle drives itself based on Lane Following model trained from the collected data
- ROS2-based
	- Time synchronous data collection node
	- deploying a trained model in a node
- Data preprocessing for training
	- Data normalization
	- Data augmentation
	- Splitting data into training set and test set
	- Writing/Reading data in HDF5 format
- Deep Learning model training: Train a model using Keras with TensorFlow backend

## Training Details

### Network Architecture

The network has 252,219 parameters and consists of 9 layers, including 5 convolutional layers, 3 fully connected layers, and an output layer.

| Layer (type) | Output Shape | Param # |
|:------------:|:------------:|:-------:|
| conv2d_1 (Conv2D) | (None, 31, 98, 24) | 1824 |
| conv2d_2 (Conv2D) | (None, 14, 47, 36) | 21636 |
| conv2d_3 (Conv2D) | (None, 5, 22, 48) | 43248 |
| conv2d_4 (Conv2D) | (None, 3, 20, 64) | 27712 |
| conv2d_5 (Conv2D) | (None, 1, 18, 64) | 36928 |
| flatten_1 (Flatten) | (None, 1152) | 0 |
| dense_1 (Dense) | (None, 100) | 115300 |
| dense_2 (Dense) | (None, 50) | 5050 |
| dense_3 (Dense) | (None, 10) | 510 |
| dense_4 (Dense) | (None, 1) | 11 |

### Hyperparameters

- Learning rate: 1e-04
- Learning rate decay: None
- Dropout: 0.5
- Mini-batch size: 256
- Epochs: 30
- Optimizer algorithm: Adam
- Loss function: Mean squared error
- Training/Test set ratio: 8:2

### Dataset
[Data image goes here]

## How to Collect Data and Train Your Own Model with LGSVL Simulator

### Collect data from LGSVL Simulator

To collect and save camera images and steering commands

`docker-compose up collect`

### Data preprocessing

`docker-compose up preprocess`

### Train a model

We use Keras with TensorFlow backend for training our model.

`docker-compose up train`

### Drive with your trained model in LGSVL Simulator

Place your trained model in `lanefollowing/ros2_ws/src/lane_following/model/model.h5` and run following to start driving:

`docker-compose up drive`

## Future Works and Contributing

Though the network can successfully drive and follow lanes on the bridge, there's still a lot of room for future improvements (biased to drive straight, afraid of shadows, few training data, and etc).
- To improve model robustness collect more training data by driving in a wide variety of environments
	- Changing weather and lighting effects (rain, fog, road wetness, time of day)
	- Adding more road layouts and road textures
	- Adding more shadows on roads
	- Adding NPC cars around the ego vehicle
- Predict the car throttle along with the steering angle
- Take into accounts time series analysis using RNN (Recurrent Neural Network)

## Links

- [Lane Following Github Repository](https://github.com/lgsvl/lanefollowing)
- [LGSVL Simulator](https://www.lgsvlsimulator.com/)
- [NVIDIA's End-to-End Deep Learning Model for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

## Copyright and License

Copyright (c) 2018 LG Electronics, Inc.

This software contains code licensed as described in LICENSE.