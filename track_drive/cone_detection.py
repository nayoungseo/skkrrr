#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32MultiArray


class ConeDetectionNode(Node):
    def __init__(self):
        super().__init__('cone_detection')
        self.get_logger().info('===== Cone Detection Node Started =====')

        self.img_w = 640
        self.img_h = 480
        self.debug = True

        self.lane_success_count = 0
        self.lane_active = False

        self.min_x = -4.0
        self.max_x = 4.0
        self.min_y = -3.5
        self.max_y = 10.5

        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)
        self.sub_lane_valid = self.create_subscription(
            Bool, '/vision/lane_valid', self.lane_valid_callback, 10)

        self.fit_x_pub = self.create_publisher(
            Float32MultiArray, '/vision/lane_fit_x', 10)
        self.valid_pub = self.create_publisher(Bool, '/vision/cone_valid', 10)

        self.last_poly = np.array([0.0, 320.0])

    def publish_valid(self, valid):
        valid_msg = Bool()
        valid_msg.data = valid
        self.valid_pub.publish(valid_msg)

    def lane_valid_callback(self, msg):
        if self.lane_active:
            return

        if msg.data:
            self.lane_success_count += 1
            if self.lane_success_count >= 8:
                self.get_logger().info(
                    'Lane detected for 8 consecutive frames. '
                    'Switching to lane driving mode.'
                )
                self.lane_active = True
                self.publish_valid(False)
                if self.debug:
                    cv2.destroyAllWindows()
        else:
            self.lane_success_count = 0

    def lidar_callback(self, msg):
        if self.lane_active:
            # Keep publishing the handoff state so late subscribers cannot
            # mistake the previous cone-valid message for an active cone mode.
            self.publish_valid(False)
            return

        left_points = []
        for index, distance in enumerate(msg.ranges):
            if not math.isfinite(distance) or distance < 1.0 or distance > 12.0:
                continue

            angle = np.deg2rad(index - 90)
            lidar_x = -distance * np.cos(angle)
            lidar_y = -distance * np.sin(angle)

            if not (
                self.min_x <= lidar_x <= self.max_x and
                self.min_y <= lidar_y <= self.max_y
            ):
                continue

            pixel_x = int(
                (lidar_x - self.min_x) /
                (self.max_x - self.min_x) *
                (self.img_w - 1)
            )
            pixel_y = int(
                (1.0 - (lidar_y - self.min_y) / (self.max_y - self.min_y)) *
                (self.img_h - 1)
            )
            pixel_x = np.clip(pixel_x, 0, self.img_w - 1)
            pixel_y = np.clip(pixel_y, 0, self.img_h - 1)

            if pixel_x >= 320 or pixel_y < 120 or -0.4 < lidar_x < 0.4:
                continue

            left_points.append([pixel_x, pixel_y, lidar_y])

        self.process_cone_lanes(left_points)

    def process_cone_lanes(self, left_points):
        plot_y = np.linspace(0, self.img_h - 1, self.img_h)

        if self.debug:
            debug_img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
            cv2.rectangle(debug_img, (315, 355), (325, 365), (0, 0, 255), -1)
            cv2.line(debug_img, (320, 360), (320, 200), (0, 0, 255), 1)

        final_fit_x = None
        valid = False
        enough_cones = False
        if len(left_points) >= 2:
            points = np.array(left_points)
            distance_spread = np.max(points[:, 2]) - np.min(points[:, 2])
            enough_cones = distance_spread >= 1.0

        if enough_cones:
            points = np.array(left_points)
            poly = np.polyfit(points[:, 1], points[:, 0], 1)
            self.last_poly = poly
            left_fit_x = poly[0] * plot_y + poly[1]

            if self.debug:
                for point in left_points:
                    cv2.circle(
                        debug_img, (int(point[0]), int(point[1])), 4,
                        (0, 255, 255), -1)
                cone_fit_x = np.clip(left_fit_x, 0, self.img_w - 1)
                cone_points = np.vstack((cone_fit_x, plot_y)).astype(np.int32).T
                cv2.polylines(
                    debug_img, [cone_points], False, (255, 255, 0), 2)
        else:
            left_fit_x = self.last_poly[0] * plot_y + self.last_poly[1]
            if self.debug:
                for point in left_points:
                    cv2.circle(
                        debug_img, (int(point[0]), int(point[1])), 4,
                        (0, 255, 255), -1)
                cv2.putText(
                    debug_img,
                    'CONE LACK (<2) -> KEEP LAST PLOT MODE',
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

        target_pixel_y = 360.0
        current_x_at_target = (
            self.last_poly[0] * target_pixel_y + self.last_poly[1]
            if not enough_cones
            else poly[0] * target_pixel_y + poly[1]
        )
        lane_offset = 320.0 - current_x_at_target
        final_fit_x = left_fit_x + lane_offset
        valid = True

        final_fit_x = np.clip(final_fit_x, 0, self.img_w - 1)
        # Publish the mode before the shared fit topic so the driver cannot
        # briefly interpret a cone fit as a camera-lane fit.
        self.publish_valid(valid)
        fit_x_msg = Float32MultiArray()
        fit_x_msg.data = final_fit_x.tolist()
        self.fit_x_pub.publish(fit_x_msg)

        if self.debug:
            center_points = np.vstack((final_fit_x, plot_y)).astype(np.int32).T
            cv2.polylines(debug_img, [center_points], False, (0, 255, 0), 2)
            cv2.putText(
                debug_img,
                'LEFT-CONE MODE (CYAN=CONE LINE, GREEN=DRIVE LINE)',
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.imshow('CONE DETECTION DEBUG', debug_img)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ConeDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
