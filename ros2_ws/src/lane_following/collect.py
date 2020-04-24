import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import TwistStamped
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

        self.get_logger().info('[{}] Initializing...'.format(self.get_name()))

        mkdir_p(CSV_PATH)
        mkdir_p(IMG_PATH)

        sub_center_camera = message_filters.Subscriber(self, CompressedImage, '/simulator/sensor/camera/center/compressed')
        sub_left_camera = message_filters.Subscriber(self, CompressedImage, '/simulator/sensor/camera/left/compressed')
        sub_right_camera = message_filters.Subscriber(self, CompressedImage, '/simulator/sensor/camera/right/compressed')
        sub_control = message_filters.Subscriber(self, TwistStamped, '/simulator/control/command')

        ts = message_filters.ApproximateTimeSynchronizer([sub_center_camera, sub_left_camera, sub_right_camera, sub_control], 1, 0.1)
        ts.registerCallback(self.callback)

        self.get_logger().info('[{}] Up and running...'.format(self.get_name()))

    def callback(self, center_camera, left_camera, right_camera, control):
        ts_sec = center_camera.header.stamp.sec
        ts_nsec = center_camera.header.stamp.nanosec
        steer_cmd = control.twist.angular.x

        self.get_logger().info("[{}.{}] Format: {}, Steering_cmd: {}".format(ts_sec, ts_nsec, center_camera.format, steer_cmd))

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
