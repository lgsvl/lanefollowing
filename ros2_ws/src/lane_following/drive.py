import rclpy
from rclpy.node import Node
from rclpy.timer import WallTimer
from sensor_msgs.msg import CompressedImage
from lgsvl_msgs.msg import VehicleControlData
import threading
import numpy as np
import cv2
import os
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from train.utils import preprocess_image
import math
import time
import argparse


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


class Drive(Node):
    def __init__(self):
        super().__init__('drive', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.log = self.get_logger()
        self.log.info('Starting Drive node...')

        self.image_lock = threading.RLock()

        # ROS topics
        self.camera_topic = self.get_param('camera_topic')
        self.control_topic = self.get_param('control_topic')

        self.log.info('Camera topic: {}'.format(self.camera_topic))
        self.log.info('Control topic: {}'.format(self.control_topic))

        # ROS communications
        self.image_sub = self.create_subscription(CompressedImage, self.camera_topic, self.image_callback)
        self.control_pub = self.create_publisher(VehicleControlData, self.control_topic)

        # ROS timer
        self.publish_period = .02  # seconds
        self.check_timer = self.create_timer(1, self.check_camera_topic)

        # ROS parameters
        self.enable_visualization = self.get_param('visualization', False)
        self.model_path = self.get_param('model_path')

        # Model parameters
        self.model = self.get_model(self.model_path)
        self.img = None
        self.steering = 0.
        self.inference_time = 0.

        # For visualizations
        self.steer_ratio = self.get_param('steer_ratio', 16.)
        self.steering_wheel_single_direction_max = self.get_param('steering_wheel_single_direction_max', 470.)  # in degrees
        self.wheel_base = self.get_param('wheel_base', 2.836747)  # in meters

        # FPS
        self.last_time = time.time()
        self.frames = 0
        self.fps = 0.

        self.log.info('Up and running...')

    def image_callback(self, img):
        self.get_fps()
        if self.image_lock.acquire(True):
            self.img = img
            if self.model is None:
                self.model = self.get_model(self.model_path)
            t0 = time.time()
            self.steering = self.predict(self.model, self.img)
            t1 = time.time()
            self.inference_time = t1 - t0
            if self.enable_visualization:
                self.visualize(self.img, self.steering)
            self.image_lock.release()

    def publish_steering(self):
        if self.img is None:
            return
        message = VehicleControlData()
        message.target_wheel_angular_rate = float(self.steering)
        self.control_pub.publish(message)
        self.log.info('[{:.3f}] Predicted steering command: "{}"'.format(time.time(), message.target_wheel_angular_rate))

    def get_model(self, model_path):
        self.log.info('Loading model from {}'.format(model_path))
        model = load_model(model_path)
        self.log.info('Model loaded!')

        return model

    def predict(self, model, img):
        c = np.fromstring(bytes(img.data), np.uint8)
        img = cv2.imdecode(c, cv2.IMREAD_COLOR)
        img = preprocess_image(img)
        img = np.expand_dims(img, axis=0)  # img = img[np.newaxis, :, :]
        steering = self.model.predict(img)

        return steering

    def visualize(self, img, steering):
        c = np.fromstring(bytes(img.data), np.uint8)
        image = cv2.imdecode(c, cv2.IMREAD_COLOR)

        steering_wheel_angle_deg = steering * self.steering_wheel_single_direction_max
        wheel_angle_deg = steering_wheel_angle_deg / self.steer_ratio  # wheel angle in degree [-29.375, 29.375]
        curvature_radius = self.wheel_base / (2 - 2 * math.cos(2 * steering_wheel_angle_deg / self.steer_ratio)) ** 2

        kappa = 1 / curvature_radius
        curvature = int(kappa * 50)
        if steering < 0:  # Turn left
            x = -curvature
            ra = 0
            rb = -70
        else:  # Turn right
            x = curvature
            ra = -110
            rb = -180

        cv2.ellipse(image, (960 + x, image.shape[0]), (curvature, 500), 0, ra, rb, (0, 255, 0), 2)

        cv2.putText(image, "Prediction: %f.7" % (steering), (30, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.putText(image, "Steering wheel angle: %.3f degrees" % steering_wheel_angle_deg, (30, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.putText(image, "Wheel angle: %.3f degrees" % wheel_angle_deg, (30, 170), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.putText(image, "Prediction time: %d ms" % (self.inference_time * 1000), (30, 220), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.putText(image, "Frame speed: %d fps" % (self.fps), (30, 270), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        image = cv2.resize(image, (round(image.shape[1] / 2), round(image.shape[0] / 2)), interpolation=cv2.INTER_AREA)
        cv2.imshow('LGSVL End-to-End Lane Following', image)
        cv2.waitKey(1)

    def get_fps(self):
        self.frames += 1
        now = time.time()
        if now >= self.last_time + 1.0:
            delta = now - self.last_time
            self.last_time = now
            self.fps = self.frames / delta
            self.frames = 0

    def get_param(self, key, default=None):
        val = self.get_parameter(key).value
        if val is None:
            val = default
        return val

    def check_camera_topic(self):
        if self.img is None:
            self.log.info('Waiting a camera image from ROS topic: {}'.format(self.camera_topic))
        else:
            self.log.info('Received a camera image from ROS topic: {}'.format(self.camera_topic))
            self.destroy_timer(self.check_timer)
            self.create_timer(self.publish_period, self.publish_steering)


def main(args=None):
    rclpy.init(args=args)
    drive = Drive()
    rclpy.spin(drive)


if __name__ == '__main__':
    main()
