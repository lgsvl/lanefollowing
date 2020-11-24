import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from lgsvl_msgs.msg import CanBusData
import message_filters
from datetime import datetime
import os
import errno
import numpy as np
import csv
import cv2
from train.utils import mkdir_p, CSV_PATH, IMG_PATH


class Collect(Node):
    def __init__(self):
        super().__init__('collect', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.log = self.get_logger()
        self.log.info('Starting Collect node...')

        mkdir_p(CSV_PATH)
        mkdir_p(IMG_PATH)

        center_camera_topic = self.get_parameter('center_camera_topic').value
        left_camera_topic = self.get_parameter('left_camera_topic').value
        right_camera_topic = self.get_parameter('right_camera_topic').value
        canbus_topic = self.get_parameter('canbus_topic').value

        self.log.info('Center camera topic: {}'.format(center_camera_topic))
        self.log.info('Left camera topic: {}'.format(left_camera_topic))
        self.log.info('Right camera topic: {}'.format(right_camera_topic))
        self.log.info('Canbus topic: {}'.format(canbus_topic))

        sub_center_camera = message_filters.Subscriber(self, CompressedImage, center_camera_topic)
        sub_left_camera = message_filters.Subscriber(self, CompressedImage, left_camera_topic)
        sub_right_camera = message_filters.Subscriber(self, CompressedImage, right_camera_topic)
        sub_canbus = message_filters.Subscriber(self, CanBusData, canbus_topic)

        ts = message_filters.ApproximateTimeSynchronizer([sub_center_camera, sub_left_camera, sub_right_camera, sub_canbus], 1, 0.1)
        ts.registerCallback(self.callback)

        self.log.info('Up and running...')

    def callback(self, center_camera, left_camera, right_camera, canbus):
        ts_sec = center_camera.header.stamp.sec
        ts_nsec = center_camera.header.stamp.nanosec
        steer_cmd = canbus.steer_pct

        self.log.info("[{}.{}] Format: {}, Steering_cmd: {}".format(ts_sec, ts_nsec, center_camera.format, steer_cmd))

        msg_id = str(datetime.now().isoformat())
        self.save_image(center_camera, left_camera, right_camera, msg_id)
        self.save_csv(steer_cmd, msg_id)

    def save_image(self, center_camera, left_camera, right_camera, msg_id):
        center_img_np_arr = np.fromstring(bytes(center_camera.data), np.uint8)
        left_img_np_arr = np.fromstring(bytes(left_camera.data), np.uint8)
        right_img_np_arr = np.fromstring(bytes(right_camera.data), np.uint8)
        center_img_cv = cv2.imdecode(center_img_np_arr, cv2.IMREAD_COLOR)
        left_img_cv = cv2.imdecode(left_img_np_arr, cv2.IMREAD_COLOR)
        right_img_cv = cv2.imdecode(right_img_np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(IMG_PATH, 'center-{}.jpg'.format(msg_id)), center_img_cv)
        cv2.imwrite(os.path.join(IMG_PATH, 'left-{}.jpg'.format(msg_id)), left_img_cv)
        cv2.imwrite(os.path.join(IMG_PATH, 'right-{}.jpg'.format(msg_id)), right_img_cv)

    def save_csv(self, steer_cmd, msg_id):
        with open(os.path.join(CSV_PATH, 'training_data.csv'), 'a+') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([msg_id, steer_cmd])


def main(args=None):
    rclpy.init(args=args)
    collect = Collect()
    rclpy.spin(collect)


if __name__ == '__main__':
    main()
