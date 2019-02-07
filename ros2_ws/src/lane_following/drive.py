import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import CompressedImage
import threading
import numpy as np
import cv2
import os
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from train.utils import preprocess_image


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_FILE = '{}/model/model.h5'.format(BASE_PATH)


class Drive(Node):
    def __init__(self):
        super().__init__('drive')

        self.image_lock = threading.RLock()
        self.image_sub = self.create_subscription(CompressedImage, '/simulator/sensor/camera/center/compressed', self.image_callback)
        self.control_pub = self.create_publisher(TwistStamped, '/lgdrive/steering_cmd')
        timer_period = .02  # seconds
        self.timer = self.create_timer(timer_period, self.publish_steering)

        self.model = None
        self.img = None
        self.steering = 0.

    def image_callback(self, img):
        if self.image_lock.acquire(True):
            self.img = img
            if self.model is None:
                self.model = self.get_model(MODEL_FILE)
            self.steering = self.predict(self.model, self.img)
            self.image_lock.release()
    
    def publish_steering(self):
        if self.img is None:
            return
        message = TwistStamped()
        message.twist.angular.x = float(self.steering)
        self.control_pub.publish(message)
        self.get_logger().info('Publishing steering cmd: "{}"'.format(message.twist.angular.x))
    
    def get_model(self, model_file):
        model = load_model(model_file)
        self.get_logger().info('Model loaded: {}'.format(model_file))

        return model
    
    def predict(self, model, img):
        c = np.fromstring(bytes(img.data), np.uint8)
        img = cv2.imdecode(c, cv2.IMREAD_COLOR)
        img = preprocess_image(img)
        img = np.expand_dims(img, axis=0)  # img = img[np.newaxis, :, :]
        steering = self.model.predict(img)

        return steering


def main(args=None):
    rclpy.init(args=args)
    drive = Drive()
    rclpy.spin(drive)


if __name__ == '__main__':
    main()
