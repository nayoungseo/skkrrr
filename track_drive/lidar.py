#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


class LidarProcessorNode(Node):
    def __init__(self):
        super().__init__('lidar_processor_node')

        self.lidar_ranges = None
        self.angle_min = 0.0
        self.angle_increment = math.radians(1.0)
        self.lane_valid_status = False
        self.started_at = time.monotonic()
        self.obstacle_detection_delay = 15.0

        self.sub_lidar = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        self.sub_lane_valid = self.create_subscription(
            Bool, '/vision/lane_valid', self.lane_valid_callback, 10)

        self.pub_obstacle = self.create_publisher(
            String, '/lidar/obstacle_status', 10)

        self.obstacle_msg = String()

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info(
            "Lidar Processor Node Started (obstacle free-space mode)")

    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment

    def lane_valid_callback(self, msg):
        self.lane_valid_status = msg.data

    def timer_callback(self):
        if self.lidar_ranges is None or len(self.lidar_ranges) == 0:
            self.get_logger().warn("No LiDAR data yet...")
            return

        ranges = self.lidar_ranges
        if time.monotonic() - self.started_at < self.obstacle_detection_delay:
            self.obstacle_msg.data = "NONE"
            self.pub_obstacle.publish(self.obstacle_msg)
            left = self.obstacle_detection_delay - (time.monotonic() - self.started_at)
            print(
                f"[lidar] DETECTION_OFF {left:4.1f}s obstacle_status=NONE",
                end='\r',
                flush=True,
            )
            return
        self.process_obstacle_calculations(ranges)

    def scan_to_xy_points(self, ranges, min_range=0.15, max_range=8.0):
        ranges_np, angles = self.scan_relative_angles_deg(ranges)
        if ranges_np.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        valid = (
            np.isfinite(ranges_np) &
            (ranges_np > min_range) &
            (ranges_np < max_range)
        )

        if not np.any(valid):
            return np.empty((0, 2), dtype=np.float32)

        valid_ranges = ranges_np[valid]
        valid_angles = np.deg2rad(angles[valid])
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        return np.column_stack((x, y)).astype(np.float32)

    def nearest_path_distance(self, points, center_y, half_width, max_x):
        if points.size == 0:
            return max_x

        in_path = (
            (points[:, 0] > 0.15) &
            (points[:, 0] < max_x) &
            (np.abs(points[:, 1] - center_y) < half_width)
        )

        if not np.any(in_path):
            return max_x

        return float(np.min(points[in_path, 0]))

    def scan_relative_angles_deg(self, ranges):
        ranges_np = np.asarray(ranges, dtype=np.float32)
        if ranges_np.size == 0:
            return ranges_np, ranges_np

        # The bundled lidar viewer maps index 0/359 to the vehicle's forward
        # direction. Negative angle is left, positive angle is right.
        indices = np.arange(ranges_np.size, dtype=np.float32)
        angles = -indices * (360.0 / ranges_np.size)
        angles = ((angles + 180.0) % 360.0) - 180.0
        return ranges_np, angles

    def corridor_stats(self, points, y_min, y_max, max_x):
        if points.size == 0:
            return max_x, 0, 0.0

        selected = points[
            (points[:, 0] > 0.35) &
            (points[:, 0] < max_x) &
            (points[:, 1] >= y_min) &
            (points[:, 1] <= y_max)
        ]

        if selected.size == 0:
            return max_x, 0, 0.0

        clear = float(np.percentile(selected[:, 0], 20.0))
        near = selected[selected[:, 0] <= clear + 0.45]
        obstacle_y = float(np.median(near[:, 1])) if near.size > 0 else 0.0
        return clear, int(selected.shape[0]), obstacle_y

    def sector_distance(
        self,
        ranges_np,
        angles,
        angle_min,
        angle_max,
        min_range=0.35,
        max_range=5.5,
        percentile=20.0,
    ):
        sector = (
            (angles >= angle_min) &
            (angles <= angle_max) &
            np.isfinite(ranges_np) &
            (ranges_np > min_range) &
            (ranges_np < max_range)
        )

        values = ranges_np[sector]
        if values.size == 0:
            return max_range, 0

        return float(np.percentile(values, percentile)), int(values.size)

    def process_obstacle_calculations(self, ranges):
        ranges_np, angles = self.scan_relative_angles_deg(ranges)
        points = self.scan_to_xy_points(ranges, min_range=0.35, max_range=5.8)

        detection_x = 5.8
        stop_x = 0.9
        min_pass_x = 1.5
        min_path_points = 3

        front_clear, front_count = self.sector_distance(
            ranges_np, angles, -8.0, 8.0, max_range=detection_x)
        path_clear, path_count, obstacle_y = self.corridor_stats(
            points, -1.05, 1.05, detection_x)
        left_clear, left_count, _ = self.corridor_stats(
            points, -1.65, -0.25, detection_x)
        right_clear, right_count, _ = self.corridor_stats(
            points, 0.25, 1.65, detection_x)

        path_blocked = path_count >= min_path_points and path_clear < detection_x
        no_escape = left_clear < min_pass_x and right_clear < min_pass_x

        if path_clear < stop_x and no_escape:
            status_str = "STOP"
        elif path_blocked:
            status_str = "AVOID_RIGHT" if obstacle_y <= 0.0 else "AVOID_LEFT"
        else:
            status_str = "NONE"

        print(
            f"[lidar] {status_str:<11} "
            f"front(-8~8)={front_clear:4.2f}m/{front_count:02d} "
            f"path={path_clear:4.2f}m/{path_count:02d}/y={obstacle_y:+.2f} "
            f"left={left_clear:4.2f}m/{left_count:02d} "
            f"right={right_clear:4.2f}m/{right_count:02d}",
            end='\r',
            flush=True,
        )

        self.obstacle_msg.data = status_str
        self.pub_obstacle.publish(self.obstacle_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nNode shutting down.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
