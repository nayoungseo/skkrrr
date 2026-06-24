#!/usr/bin/env python3
# -*- coding: utf-8 -*- 1
#=============================================
# 본 프로그램은 자이트론에서 제작한 것입니다.
# 상업라이센스에 의해 제공되므로 무단배포 및 상업적 이용을 금합니다.
# 교육과 실습 용도로만 사용가능하며 외부유출은 금지됩니다.
#=============================================
import rclpy, time, cv2, os, math, json
import numpy as np
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from cv_bridge import CvBridge

# [추가] 외부 차선 인식 노드의 토픽을 받기 위한 ROS2 표준 메시지
from std_msgs.msg import Float32MultiArray, Bool, String

#=============================================
# ROS2 Node 클래스 정의
#=============================================
class TrackDriverNode(Node):

    #=============================================
    # 클래스 생성 초기화 함수
    #=============================================
    def __init__(self):

        super().__init__('driver')
        self.get_logger().info('----- Xycar self-driving node started -----')
        
        # 상수값 및 초기값 설정
        self.image = None  # 카메라 토픽 데이터를 저장할 변수
        self.motor_msg = XycarMotor()  # 모터토픽 메시지        
        self.lidar_ranges = None
        self.latest_scan = None
        self.bridge = CvBridge()
        self.last_fit_x_time = 0.0
        self.last_fit_x = None
        self.last_camera_time = 0.0
        self.last_scan_time = 0.0
        self.last_cmd_angle = 0.0
        self.last_cmd_speed = 0.0
        self.drive_started_at = time.monotonic()
        self.avoidance_ignore_duration = 15.0

        # 차량 내부 이미지 크기 기본 규격 정의 (Sliding window 이미지 크기 기준)
        self.img_w = 640
        self.img_h = 480
        
        # ROS2 Publisher & Subscriber 설정
        self.motor_pub = self.create_publisher(XycarMotor,'xycar_motor',10)
        self.avoidance_debug_pub = self.create_publisher(
            String,
            '/track_drive/avoidance_debug',
            10,
        )
        
        self.sub_front = self.create_subscription(
            Image, '/usb_cam/image_raw/front', self.cam_callback, qos_profile_sensor_data)

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)
		
        # [수정] /vision/lane_fit_x 토픽을 Float32MultiArray 데이터형으로 구독하도록 변경
        self.sub_fit_x = self.create_subscription(
            Float32MultiArray, '/vision/lane_fit_x', self.lane_fit_x_callback, 10)
            
        self.sub_valid = self.create_subscription(
            Bool, '/vision/lane_valid', self.lane_valid_callback, 10)

        self.sub_cone_valid = self.create_subscription(
            Bool, '/vision/cone_valid', self.cone_valid_callback, 10)

        # --------------------------------------------------------
        # [신규 추가] 라이다 장애물 상태 토픽 구독 설정
        # --------------------------------------------------------
        self.sub_obstacle = self.create_subscription(
            String, '/lidar/obstacle_status', self.obstacle_status_callback, 10)
        self.debug_timer = self.create_timer(1.0, self.debug_status_timer)

        # --------------------------------------------------------
        # [신규 추가] 상태 제어 변수 및 모듈별 퍼블리셔 초기화
        # --------------------------------------------------------
        self.prev_angle = 0.0  # 차선 소실 시 유지할 이전 조향각 백업 변수
        self.current_state = "STRAIGHT" # 초기 차량 상태 (STRAIGHT / CURVE / EMERGENCY)
        self.lane_valid_status = True
        self.cone_valid_status = False

        # [신규 변수] 실시간 라이다 회피 상태 및 변이 오프셋 저장 변수
        self.obstacle_state = "NONE"
        self.obstacle_offset = 0.0
        self.obstacle_target_offset = 0.0
        self.obstacle_hold_until = 0.0
        self.obstacle_speed_cap = None
        self.right_avoid_offset = -100.0
        self.left_avoid_offset = 100.0
        self.dynamic_stage = "IDLE"
        self.last_reported_dynamic_stage = "IDLE"
        self.dynamic_clear_start = None
        self.dynamic_cooldown_until = 0.0
        self.dynamic_shift_right_duration = 1.5
        self.dynamic_shift_left_duration = 2.5
        self.dynamic_shift_timeout_extra = 0.9
        self.dynamic_shift_settle_tolerance_px = 34.0
        self.dynamic_heading_settle_tolerance_px = 42.0
        self.dynamic_steering_settle_tolerance_deg = 30.0
        self.dynamic_right_hold = 4.0
        self.dynamic_left_hold = 7.0
        self.dynamic_shift_kp = 0.075
        self.dynamic_hold_kp = 0.052
        self.lidar_clear_hold = 0.35
        self.dynamic_stage_started_at = 0.0
        self.camera_pedestrian_seen_until = 0.0
        self.lidar_pedestrian_seen_until = 0.0
        self.pedestrian_target_offset = 0.0
        self.last_pedestrian_check = 0.0
        self.pedestrian_clear_start = None
        self.pedestrian_stop_required = False
        self.pedestrian_min_stop_duration = 3.0
        self.pedestrian_stuck_timeout = 5.0
        self.pedestrian_backup_duration = 0.8
        self.pedestrian_lidar_only_ignore_until = 0.0
        self.pedestrian_source = "none"
        self.planner_last_angle = 0.0
        self.obstacle_angle_override = None
        self.last_lidar_pedestrian_check = 0.0
        self.left_lidar_track = None
        self.left_lidar_seen_until = 0.0
        self.left_lidar_status = "none"
        self.left_lidar_closure_rate = None
        self.left_lidar_distance = None
        self.left_lidar_min_dt = 0.20
        self.left_lidar_closure_threshold = 0.35
        self.car_trigger_mode = "none"
        self.dynamic_lane_error_px = None
        self.dynamic_lane_heading_error_px = None

        self.straight_consecutive_count = 0

        self.get_logger().info("Track Driver Node Initialized")

    def obstacle_status_callback(self, msg):
        if time.monotonic() - self.drive_started_at < self.avoidance_ignore_duration:
            self.obstacle_state = "NONE"
            return
        self.obstacle_state = msg.data.strip().upper()

    def debug_status_timer(self):
        now = time.monotonic()
        fit_age = now - self.last_fit_x_time if self.last_fit_x_time > 0.0 else None
        cam_age = now - self.last_camera_time if self.last_camera_time > 0.0 else None
        scan_age = now - self.last_scan_time if self.last_scan_time > 0.0 else None
        motor_publishers = self.count_publishers('xycar_motor')
        motor_subscribers = self.count_subscribers('xycar_motor')
        detection_ignore_left = max(
            0.0,
            self.avoidance_ignore_duration - (now - self.drive_started_at)
        )

        if motor_publishers > 1:
            self.get_logger().warn(
                f"xycar_motor publishers={motor_publishers}. "
                "Another node may overwrite this driver's motor command."
            )

        if motor_subscribers == 0:
            self.get_logger().warn(
                "xycar_motor subscribers=0. The simulator is not receiving motor commands."
            )

        if fit_age is None or fit_age > 1.0:
            self.get_logger().warn(
                "No recent /vision/lane_fit_x; autonomous drive callback is not running."
            )

        self.get_logger().info(
            "driver status: "
            f"cmd=({self.last_cmd_angle:.1f},{self.last_cmd_speed:.1f}) "
            f"motor_pub={motor_publishers} motor_sub={motor_subscribers} "
            f"lane_valid={self.lane_valid_status} "
            f"cone_valid={self.cone_valid_status} "
            f"state={self.current_state} mission={self.dynamic_stage} "
            f"obs={self.obstacle_state} offset={self.obstacle_offset:.1f} "
            f"car_trigger={self.car_trigger_mode} "
            f"left_lidar={self.left_lidar_status} "
            f"detect_ignore={detection_ignore_left:.1f}s "
            f"fit_age={fit_age if fit_age is not None else -1:.2f}s "
            f"cam_age={cam_age if cam_age is not None else -1:.2f}s "
            f"scan_age={scan_age if scan_age is not None else -1:.2f}s"
        )

    def dynamic_stage_hold_seconds(self):
        if self.dynamic_stage == "DYN_SHIFT_RIGHT":
            return self.dynamic_shift_right_duration
        if self.dynamic_stage == "DYN_HOLD_RIGHT":
            return self.dynamic_right_hold
        if self.dynamic_stage == "DYN_SHIFT_LEFT":
            return self.dynamic_shift_left_duration
        if self.dynamic_stage == "DYN_HOLD_LEFT":
            return self.dynamic_left_hold
        if self.dynamic_stage == "PED_STOP":
            return self.pedestrian_min_stop_duration
        if self.dynamic_stage == "PED_BACKUP":
            return self.pedestrian_backup_duration
        return None

    def is_dynamic_car_stage(self):
        return self.dynamic_stage in (
            "DYN_SHIFT_RIGHT",
            "DYN_HOLD_RIGHT",
            "DYN_SHIFT_LEFT",
            "DYN_HOLD_LEFT",
            "DYN_RETURN_CENTER",
        )

    def is_dynamic_shift_stage(self):
        return self.dynamic_stage in (
            "DYN_SHIFT_RIGHT",
            "DYN_SHIFT_LEFT",
        )

    def is_dynamic_lane_tracking_stage(self):
        return self.dynamic_stage in (
            "DYN_SHIFT_RIGHT",
            "DYN_HOLD_RIGHT",
            "DYN_SHIFT_LEFT",
            "DYN_HOLD_LEFT",
            "DYN_RETURN_CENTER",
        )

    def dynamic_lane_tracking_offset(self):
        if self.dynamic_stage in (
            "DYN_SHIFT_RIGHT",
            "DYN_HOLD_RIGHT",
            "DYN_SHIFT_LEFT",
            "DYN_HOLD_LEFT",
            "DYN_RETURN_CENTER",
        ):
            return float(self.obstacle_offset)
        return 0.0

    def lane_target_error_px(self, offset, look_ahead_ratio=0.62):
        fit_x = self.last_fit_x
        if fit_x is None or len(fit_x) == 0:
            self.dynamic_lane_error_px = None
            self.dynamic_lane_heading_error_px = None
            return None

        idx = int(np.clip(self.img_h * look_ahead_ratio, 0, len(fit_x) - 1))
        far_idx = int(np.clip(self.img_h * 0.52, 0, len(fit_x) - 1))
        near_idx = int(np.clip(self.img_h * 0.82, 0, len(fit_x) - 1))
        target_center = self.img_w * 0.5 + offset
        error = float(fit_x[idx] - target_center)
        heading_error = float(fit_x[near_idx] - fit_x[far_idx])
        self.dynamic_lane_error_px = error
        self.dynamic_lane_heading_error_px = heading_error
        return error

    def dynamic_shift_target_reached(self, offset):
        error = self.lane_target_error_px(offset)
        heading_error = self.dynamic_lane_heading_error_px
        if error is None or heading_error is None:
            return False
        position_ok = abs(error) <= self.dynamic_shift_settle_tolerance_px
        heading_ok = (
            abs(heading_error) <= self.dynamic_heading_settle_tolerance_px
        )
        steering_ok = (
            abs(self.prev_angle) <= self.dynamic_steering_settle_tolerance_deg
        )
        return position_ok and heading_ok and steering_ok

    def publish_avoidance_debug(self, now):
        hold = self.dynamic_stage_hold_seconds()
        elapsed = (
            max(0.0, now - self.dynamic_stage_started_at)
            if self.dynamic_stage != "IDLE"
            else 0.0
        )
        payload = {
            "stage": self.dynamic_stage,
            "stage_elapsed": elapsed,
            "stage_hold": hold,
            "car_avoid_active": self.is_dynamic_car_stage(),
            "car_trigger_mode": self.car_trigger_mode,
            "left_lidar_status": self.left_lidar_status,
            "left_lidar_closure_rate": self.left_lidar_closure_rate,
            "left_lidar_distance": self.left_lidar_distance,
            "pedestrian_source": self.pedestrian_source,
            "lane_target_error_px": self.dynamic_lane_error_px,
            "lane_heading_error_px": self.dynamic_lane_heading_error_px,
            "target_offset": float(self.obstacle_target_offset),
            "current_offset": float(self.obstacle_offset),
            "speed_cap": (
                None
                if self.obstacle_speed_cap is None
                else float(self.obstacle_speed_cap)
            ),
        }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self.avoidance_debug_pub.publish(msg)

    def scan_points_with_indices(self, min_range=0.35, max_range=7.0):
        if self.lidar_ranges is None:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        ranges_np = np.asarray(self.lidar_ranges, dtype=np.float32)
        if ranges_np.size == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        indices = np.arange(ranges_np.size, dtype=np.float32)
        angles_deg = -indices * (360.0 / ranges_np.size)
        angles_deg = ((angles_deg + 180.0) % 360.0) - 180.0
        angles_rad = np.deg2rad(angles_deg)

        valid = (
            np.isfinite(ranges_np) &
            (ranges_np > min_range) &
            (ranges_np < max_range)
        )

        if not np.any(valid):
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        x = ranges_np * np.cos(angles_rad)
        y = ranges_np * np.sin(angles_rad)
        points = np.column_stack((x[valid], y[valid])).astype(np.float32)
        return points, np.nonzero(valid)[0].astype(np.int32)

    def lidar_corridor_clusters(
        self,
        min_x=0.6,
        max_x=6.8,
        max_abs_y=1.7,
        min_points=2,
    ):
        if self.lidar_ranges is None:
            return []

        ranges_np = np.asarray(self.lidar_ranges, dtype=np.float32)
        if ranges_np.size == 0:
            return []

        n = ranges_np.size
        indices = np.arange(n, dtype=np.float32)
        angles_deg = -indices * (360.0 / n)
        angles_deg = ((angles_deg + 180.0) % 360.0) - 180.0
        angles_rad = np.deg2rad(angles_deg)

        valid = (
            np.isfinite(ranges_np) &
            (ranges_np > 0.35) &
            (ranges_np < max_x + 0.8)
        )
        x = ranges_np * np.cos(angles_rad)
        y = ranges_np * np.sin(angles_rad)

        corridor = (
            valid &
            (x >= min_x) &
            (x <= max_x) &
            (np.abs(y) <= max_abs_y)
        )

        raw_indices = np.nonzero(corridor)[0]
        if raw_indices.size == 0:
            return []

        split_at = np.where(np.diff(raw_indices) > 1)[0] + 1
        groups = [group for group in np.split(raw_indices, split_at) if group.size >= min_points]

        if (
            len(groups) >= 2 and
            groups[0][0] == 0 and
            groups[-1][-1] == n - 1
        ):
            groups = [np.concatenate((groups[-1], groups[0]))] + groups[1:-1]

        clusters = []
        for group in groups:
            pts = np.column_stack((x[group], y[group])).astype(np.float32)
            if pts.shape[0] < min_points:
                continue

            clusters.append({
                "count": int(pts.shape[0]),
                "min_x": float(np.min(pts[:, 0])),
                "median_x": float(np.median(pts[:, 0])),
                "max_x": float(np.max(pts[:, 0])),
                "median_y": float(np.median(pts[:, 1])),
                "min_y": float(np.min(pts[:, 1])),
                "max_y": float(np.max(pts[:, 1])),
                "span_x": float(np.max(pts[:, 0]) - np.min(pts[:, 0])),
                "span_y": float(np.max(pts[:, 1]) - np.min(pts[:, 1])),
            })

        return clusters

    def detect_pedestrian_lidar_candidate(self):
        now = time.monotonic()

        if self.dynamic_stage.startswith("DYN"):
            return False, self.pedestrian_target_offset

        if now - self.last_lidar_pedestrian_check < 0.10:
            return now < self.lidar_pedestrian_seen_until, self.pedestrian_target_offset

        self.last_lidar_pedestrian_check = now
        clusters = self.lidar_corridor_clusters(
            min_x=0.75,
            max_x=4.6,
            max_abs_y=0.88,
            min_points=2,
        )

        vehicle_like = [
            cluster for cluster in clusters
            if (
                cluster["count"] >= 8 and
                max(cluster["span_x"], cluster["span_y"]) > 0.80
            )
        ]
        if len(vehicle_like) >= 2:
            return False, self.pedestrian_target_offset

        pedestrian_candidates = []
        for cluster in clusters:
            if cluster["count"] > 16:
                continue
            if cluster["span_x"] > 0.95 or cluster["span_y"] > 0.80:
                continue
            pedestrian_candidates.append(cluster)

        if not pedestrian_candidates:
            return now < self.lidar_pedestrian_seen_until, self.pedestrian_target_offset

        best = min(
            pedestrian_candidates,
            key=lambda cluster: (cluster["median_x"], abs(cluster["median_y"]))
        )
        obs_x = best["median_x"]
        obs_y = best["median_y"]

        left_limit = -1.08 if self.dynamic_stage == "PED_STOP" else -0.78
        right_limit = 0.78
        close_enough = (
            obs_x < 2.85 and
            left_limit <= obs_y <= right_limit
        )
        if not close_enough:
            return now < self.lidar_pedestrian_seen_until, self.pedestrian_target_offset

        if obs_y < -0.12:
            self.pedestrian_target_offset = self.right_avoid_offset
        elif obs_y > 0.12:
            self.pedestrian_target_offset = self.left_avoid_offset
        else:
            self.pedestrian_target_offset = self.right_avoid_offset

        self.pedestrian_stop_required = obs_x < 1.35 and abs(obs_y) < 0.52
        self.lidar_pedestrian_seen_until = now + 0.35
        return True, self.pedestrian_target_offset

    def is_pedestrian_like_lidar_cluster(self, cluster):
        return (
            cluster["count"] <= 16 and
            cluster["span_x"] <= 0.95 and
            cluster["span_y"] <= 0.85 and
            cluster["median_x"] <= 4.8 and
            abs(cluster["median_y"]) <= 1.15
        )

    def is_vehicle_like_lidar_cluster(self, cluster):
        extent = max(cluster["span_x"], cluster["span_y"])
        if self.is_pedestrian_like_lidar_cluster(cluster):
            return False
        if cluster["count"] >= 7 and extent >= 0.75:
            return True
        if cluster["count"] >= 12 and extent >= 0.45:
            return True
        if (
            cluster["count"] >= 8 and
            cluster["span_y"] >= 0.55 and
            cluster["span_x"] <= 1.25 and
            cluster["median_x"] > 1.0
        ):
            return True
        return False

    def vehicle_detection_allowed(self, now):
        fit_age = now - self.last_fit_x_time if self.last_fit_x_time > 0.0 else 999.0
        if self.current_state == "EMERGENCY" or not self.lane_valid_status:
            self.car_trigger_mode = "lane_unstable"
            return False
        if fit_age > 0.45:
            self.car_trigger_mode = "lane_stale"
            return False
        if self.last_fit_x is None or len(self.last_fit_x) < int(self.img_h * 0.9):
            self.car_trigger_mode = "lane_missing"
            return False

        try:
            sample_indices = [
                int(self.img_h * 0.55),
                int(self.img_h * 0.72),
                int(self.img_h * 0.88),
            ]
            xs = [float(self.last_fit_x[index]) for index in sample_indices]
        except (IndexError, TypeError, ValueError):
            self.car_trigger_mode = "lane_bad_fit"
            return False

        if any(x < self.img_w * 0.12 or x > self.img_w * 0.88 for x in xs):
            self.car_trigger_mode = "lane_off_center"
            return False
        if max(xs) - min(xs) > self.img_w * 0.46:
            self.car_trigger_mode = "lane_unstable_fit"
            return False

        return True

    def update_left_lidar_approach_track(self, distance, now):
        self.left_lidar_distance = float(distance)

        if (
            self.left_lidar_track is None or
            now - self.left_lidar_track["time"] > 0.90
        ):
            self.left_lidar_track = {
                "distance": float(distance),
                "time": now,
            }
            self.left_lidar_closure_rate = None
            self.left_lidar_status = "tracking"
            return now < self.left_lidar_seen_until

        previous_distance = self.left_lidar_track["distance"]
        previous_time = self.left_lidar_track["time"]
        dt = max(now - previous_time, 1e-3)

        if dt < self.left_lidar_min_dt:
            self.left_lidar_status = "tracking"
            return now < self.left_lidar_seen_until

        delta = float(distance) - previous_distance
        self.left_lidar_track = {
            "distance": float(distance),
            "time": now,
        }

        if abs(delta) > 1.20:
            self.left_lidar_closure_rate = None
            self.left_lidar_status = "jump"
            return now < self.left_lidar_seen_until

        closure_rate = -delta / dt
        self.left_lidar_closure_rate = float(closure_rate)
        approaching = (
            self.last_cmd_speed > 5.0 and
            float(distance) < 6.5 and
            closure_rate > self.left_lidar_closure_threshold
        )

        if approaching:
            self.left_lidar_seen_until = now + 0.65
            self.left_lidar_status = "approaching"
            return True

        self.left_lidar_status = "stable"
        return now < self.left_lidar_seen_until

    def detect_left_lidar_approaching_vehicle(self):
        now = time.monotonic()
        clusters = self.lidar_corridor_clusters(
            min_x=0.7,
            max_x=6.5,
            max_abs_y=1.45,
            min_points=3,
        )

        left_vehicle_clusters = []
        for cluster in clusters:
            if (
                -1.05 <= cluster["median_y"] <= -0.22 and
                cluster["min_x"] < 6.4 and
                self.is_vehicle_like_lidar_cluster(cluster)
            ):
                left_vehicle_clusters.append(cluster)

        if not left_vehicle_clusters:
            if now >= self.left_lidar_seen_until:
                self.left_lidar_status = "none"
                self.left_lidar_closure_rate = None
                self.left_lidar_distance = None
                self.left_lidar_track = None
            return now < self.left_lidar_seen_until

        left_vehicle = min(
            left_vehicle_clusters,
            key=lambda cluster: cluster["min_x"],
        )
        if left_vehicle["min_x"] < 4.3 and self.last_cmd_speed > 5.0:
            self.left_lidar_distance = float(left_vehicle["median_x"])
            self.left_lidar_seen_until = now + 0.65
            self.left_lidar_status = "close"
            return True

        return self.update_left_lidar_approach_track(
            left_vehicle["median_x"],
            now,
        )

    def detect_dynamic_vehicle_pair_lidar(self):
        clusters = self.lidar_corridor_clusters(
            min_x=0.6,
            max_x=6.4,
            max_abs_y=1.45,
            min_points=3,
        )

        vehicle_clusters = [
            cluster for cluster in clusters
            if self.is_vehicle_like_lidar_cluster(cluster)
        ]

        for i, first in enumerate(vehicle_clusters):
            for second in vehicle_clusters[i + 1:]:
                lateral_spread = abs(first["median_y"] - second["median_y"])
                nearest_x = min(first["min_x"], second["min_x"])
                opposite_sides = (
                    (first["median_y"] < -0.20 and second["median_y"] > 0.20) or
                    (second["median_y"] < -0.20 and first["median_y"] > 0.20)
                )
                if lateral_spread >= 0.60 and nearest_x < 6.2 and opposite_sides:
                    self.car_trigger_mode = "lidar_pair"
                    return True

        if self.detect_left_lidar_approaching_vehicle():
            self.car_trigger_mode = f"left_lidar_{self.left_lidar_status}"
            return True

        self.car_trigger_mode = "none"
        return False


    def detect_pedestrian_candidate(self):
        now = time.monotonic()

        if self.image is None:
            return False, 0.0

        if now - self.last_pedestrian_check < 0.12:
            return now < self.camera_pedestrian_seen_until, self.pedestrian_target_offset

        self.last_pedestrian_check = now
        frame = self.image
        h, w = frame.shape[:2]

        left_ratio = 0.06 if self.dynamic_stage == "PED_STOP" else 0.14
        x1 = int(w * left_ratio)
        x2 = int(w * 0.92)
        y1 = int(h * 0.20)
        y2 = int(h * 0.92)
        roi = frame[y1:y2, x1:x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        blue_green = cv2.inRange(
            hsv,
            np.array([35, 45, 35]),
            np.array([135, 255, 255])
        )
        dark = cv2.inRange(
            hsv,
            np.array([0, 0, 0]),
            np.array([180, 255, 80])
        )
        mask = cv2.bitwise_or(blue_green, dark)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 15), np.uint8))
        skin = cv2.inRange(
            hsv,
            np.array([0, 35, 70]),
            np.array([28, 210, 255])
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        best = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 150 or area > 5200:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh < 35 or bw < 8:
                continue
            if bw > w * 0.16 or bh > h * 0.55:
                continue

            aspect = bh / max(float(bw), 1.0)
            if aspect < 1.55 or aspect > 7.0:
                continue

            global_x = x + x1
            global_y = y + y1
            center_x = global_x + bw * 0.5
            bottom_y = global_y + bh

            if bottom_y < h * 0.38:
                continue
            if center_x < w * left_ratio or center_x > w * 0.92:
                continue
            if not self.is_in_pedestrian_road_roi(center_x, bottom_y, w, h):
                continue

            foot_y1 = int(np.clip(bottom_y, 0, h - 1))
            foot_y2 = int(np.clip(bottom_y + h * 0.06, foot_y1 + 1, h))
            foot_x1 = int(np.clip(center_x - bw * 0.8, 0, w - 1))
            foot_x2 = int(np.clip(center_x + bw * 0.8, foot_x1 + 1, w))
            foot_patch = frame[foot_y1:foot_y2, foot_x1:foot_x2]
            if foot_patch.size > 0:
                foot_hsv = cv2.cvtColor(foot_patch, cv2.COLOR_BGR2HSV)
                grass = cv2.inRange(
                    foot_hsv,
                    np.array([38, 45, 70]),
                    np.array([95, 255, 255])
                )
                grass_ratio = (
                    np.count_nonzero(grass) / max(float(grass.size), 1.0)
                )
                if grass_ratio > 0.45:
                    continue

            bbox_area = max(float(bw * bh), 1.0)
            upper_end = max(1, int(bh * 0.50))
            lower_start = min(bh - 1, int(bh * 0.42))
            skin_box = skin[y:y + bh, x:x + bw]
            dark_box = dark[y:y + bh, x:x + bw]
            color_box = blue_green[y:y + bh, x:x + bw]
            upper_skin = int(np.count_nonzero(skin_box[:upper_end, :]))
            lower_dark = int(np.count_nonzero(dark_box[lower_start:, :]))
            body_color = int(np.count_nonzero(color_box))
            skin_ok = upper_skin >= max(8, int(bbox_area * 0.012))
            legs_ok = lower_dark >= max(14, int(bbox_area * 0.018))
            body_ok = body_color >= max(18, int(bbox_area * 0.025))
            green_ratio = body_color / bbox_area

            if not ((skin_ok and (legs_ok or body_ok)) or (legs_ok and body_ok and aspect >= 2.0)):
                continue
            if green_ratio > 0.58 and not skin_ok:
                continue

            score = area * aspect + upper_skin * 4.0 + lower_dark * 0.6
            if score > best_score:
                best_score = score
                best = (center_x, bottom_y, area)

        if best is None:
            if now >= self.camera_pedestrian_seen_until:
                self.pedestrian_stop_required = False
            return now < self.camera_pedestrian_seen_until, self.pedestrian_target_offset

        center_x, bottom_y, area = best
        close_enough = bottom_y > h * 0.45 and area > 230
        if not close_enough:
            return now < self.camera_pedestrian_seen_until, self.pedestrian_target_offset

        if center_x < w * 0.48:
            self.pedestrian_target_offset = self.right_avoid_offset
        elif center_x > w * 0.52:
            self.pedestrian_target_offset = self.left_avoid_offset
        else:
            self.pedestrian_target_offset = self.right_avoid_offset

        self.pedestrian_stop_required = (
            bottom_y > h * 0.82 and
            abs(center_x - w * 0.5) < w * 0.16
        )
        self.camera_pedestrian_seen_until = now + 0.35
        return True, self.pedestrian_target_offset

    def is_in_front_road_roi(self, x, y, width, height):
        if y < height * 0.22 or y > height * 0.82:
            return False

        t = np.clip((y - height * 0.22) / (height * 0.60), 0.0, 1.0)
        left_bound = width * (0.36 - 0.28 * t)
        right_bound = width * (0.64 + 0.28 * t)
        return left_bound <= x <= right_bound

    def is_in_pedestrian_road_roi(self, x, y, width, height):
        if y < height * 0.24 or y > height * 0.95:
            return False

        t = np.clip((y - height * 0.24) / (height * 0.71), 0.0, 1.0)
        left_bound_ratio = 0.29 if self.dynamic_stage == "PED_STOP" else 0.34
        left_bound = width * (left_bound_ratio - 0.30 * t)
        right_bound = width * (0.68 + 0.31 * t)
        return left_bound <= x <= right_bound

    def vehicle_like_candidates_from_mask(
        self,
        mask,
        roi_x,
        roi_y,
        image_width,
        image_height,
        min_area,
        max_area,
        min_w,
        min_h,
        max_h,
    ):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < min_w or bh < min_h or bh > max_h:
                continue

            aspect = bw / max(float(bh), 1.0)
            if aspect < 0.65 or aspect > 5.0:
                continue

            fill_ratio = area / max(float(bw * bh), 1.0)
            if fill_ratio < 0.18:
                continue

            center_x = roi_x + x + bw * 0.5
            center_y = roi_y + y + bh * 0.5

            if not self.is_in_front_road_roi(
                center_x,
                center_y,
                image_width,
                image_height
            ):
                continue

            candidates.append((center_x, center_y, bw, bh, area))

        return candidates

    def detect_dynamic_vehicle_pair(self):
        return self.detect_dynamic_vehicle_pair_lidar()

    def update_obstacle_control(self):
        now = time.monotonic()
        desired_offset = 0.0
        speed_cap = None
        dynamic_vehicle_pair_active = False
        self.obstacle_angle_override = None

        if now - self.drive_started_at < self.avoidance_ignore_duration:
            self.dynamic_stage = "IDLE"
            self.dynamic_clear_start = None
            self.pedestrian_clear_start = None
            self.pedestrian_stop_required = False
            self.obstacle_target_offset = 0.0
            self.obstacle_speed_cap = None
            delta = np.clip(-self.obstacle_offset, -16.0, 16.0)
            self.obstacle_offset += float(delta)
            if abs(self.obstacle_offset) < 1.0:
                self.obstacle_offset = 0.0
            self.publish_avoidance_debug(now)
            return None

        camera_pedestrian_active, camera_pedestrian_offset = self.detect_pedestrian_candidate()
        if (
            now < self.pedestrian_lidar_only_ignore_until and
            not camera_pedestrian_active
        ):
            lidar_pedestrian_active = False
            lidar_pedestrian_offset = self.pedestrian_target_offset
        else:
            lidar_pedestrian_active, lidar_pedestrian_offset = (
                self.detect_pedestrian_lidar_candidate()
            )
        pedestrian_active = (
            camera_pedestrian_active or
            lidar_pedestrian_active
        )
        if camera_pedestrian_active and lidar_pedestrian_active:
            self.pedestrian_source = "both"
        elif camera_pedestrian_active:
            self.pedestrian_source = "camera"
        elif lidar_pedestrian_active:
            self.pedestrian_source = "lidar"
        else:
            self.pedestrian_source = "none"

        if (
            self.dynamic_stage == "IDLE" and
            not pedestrian_active and
            now > self.dynamic_cooldown_until and
            self.vehicle_detection_allowed(now)
        ):
            dynamic_vehicle_pair_active = self.detect_dynamic_vehicle_pair()
        elif pedestrian_active:
            self.car_trigger_mode = "blocked_by_ped"

        pedestrian_offset = (
            camera_pedestrian_offset
            if camera_pedestrian_active
            else lidar_pedestrian_offset
        )
        if not pedestrian_active:
            self.pedestrian_stop_required = False

        if self.dynamic_stage == "IDLE" and pedestrian_active:
            self.dynamic_stage = "PED_STOP"
            self.dynamic_stage_started_at = now
            self.pedestrian_clear_start = None

        if self.dynamic_stage == "PED_STOP":
            desired_offset = 0.0
            speed_cap = 0.0

            if now - self.dynamic_stage_started_at >= self.pedestrian_stuck_timeout:
                self.dynamic_stage = "PED_BACKUP"
                self.dynamic_stage_started_at = now
                self.pedestrian_clear_start = None
                self.pedestrian_lidar_only_ignore_until = now + 2.2
                desired_offset = 0.0
                speed_cap = -3.5
                self.obstacle_angle_override = 0.0

        if self.dynamic_stage == "PED_BACKUP":
            desired_offset = 0.0
            speed_cap = -3.5
            self.obstacle_angle_override = 0.0

            if now - self.dynamic_stage_started_at >= self.pedestrian_backup_duration:
                self.dynamic_stage = "IDLE"
                self.dynamic_cooldown_until = now + 0.5
                self.dynamic_stage_started_at = now
                self.pedestrian_clear_start = None
                self.pedestrian_stop_required = False
                speed_cap = None

        elif self.dynamic_stage == "PED_STOP":
            desired_offset = 0.0
            speed_cap = 0.0

            if pedestrian_active:
                self.pedestrian_clear_start = None
            else:
                stopped_long_enough = (
                    now - self.dynamic_stage_started_at >=
                    self.pedestrian_min_stop_duration
                )
                if stopped_long_enough:
                    self.dynamic_stage = "IDLE"
                    self.dynamic_cooldown_until = now + 0.4
                    self.dynamic_stage_started_at = now
                    self.pedestrian_clear_start = None

        elif self.dynamic_stage == "PED_AVOID":
            desired_offset = pedestrian_offset if pedestrian_active else self.pedestrian_target_offset
            speed_cap = 4.5

            if pedestrian_active and self.pedestrian_stop_required:
                self.dynamic_stage = "PED_STOP"
                self.dynamic_stage_started_at = now
                self.pedestrian_clear_start = None
            elif pedestrian_active:
                self.pedestrian_clear_start = None
            else:
                self.dynamic_stage = "PED_RETURN"
                self.dynamic_stage_started_at = now
                self.pedestrian_clear_start = None

        elif self.dynamic_stage == "PED_RETURN":
            desired_offset = 0.0
            speed_cap = 8.0

            if abs(self.obstacle_offset) < 2.0:
                self.dynamic_stage = "IDLE"
                self.dynamic_cooldown_until = now + 0.4
                speed_cap = None

        else:
            if (
                self.dynamic_stage == "IDLE" and
                now > self.dynamic_cooldown_until and
                dynamic_vehicle_pair_active
            ):
                self.dynamic_stage = "DYN_SHIFT_RIGHT"
                self.dynamic_stage_started_at = now
                self.dynamic_clear_start = None

            if self.dynamic_stage == "LIDAR_AVOID_RIGHT":
                desired_offset = self.right_avoid_offset
                speed_cap = 5.0

                if self.obstacle_state == "AVOID_LEFT":
                    self.dynamic_stage = "LIDAR_AVOID_LEFT"
                    self.dynamic_stage_started_at = now
                    self.dynamic_clear_start = None
                elif self.obstacle_state in ("AVOID_RIGHT", "STOP"):
                    self.dynamic_clear_start = None
                else:
                    if self.dynamic_clear_start is None:
                        self.dynamic_clear_start = now
                    if now - self.dynamic_clear_start >= self.lidar_clear_hold:
                        self.dynamic_stage = "DYN_RETURN_CENTER"
                        self.dynamic_stage_started_at = now
                        self.dynamic_clear_start = None

            elif self.dynamic_stage == "LIDAR_AVOID_LEFT":
                desired_offset = self.left_avoid_offset
                speed_cap = 5.0

                if self.obstacle_state == "AVOID_RIGHT":
                    self.dynamic_stage = "LIDAR_AVOID_RIGHT"
                    self.dynamic_stage_started_at = now
                    self.dynamic_clear_start = None
                elif self.obstacle_state in ("AVOID_LEFT", "STOP"):
                    self.dynamic_clear_start = None
                else:
                    if self.dynamic_clear_start is None:
                        self.dynamic_clear_start = now
                    if now - self.dynamic_clear_start >= self.lidar_clear_hold:
                        self.dynamic_stage = "DYN_RETURN_CENTER"
                        self.dynamic_stage_started_at = now
                        self.dynamic_clear_start = None

            elif self.dynamic_stage == "DYN_SHIFT_RIGHT":
                desired_offset = self.right_avoid_offset
                speed_cap = 7.8
                elapsed = now - self.dynamic_stage_started_at
                target_reached = self.dynamic_shift_target_reached(
                    self.right_avoid_offset
                )
                target_timed_out = (
                    elapsed >=
                    self.dynamic_shift_right_duration +
                    self.dynamic_shift_timeout_extra
                )

                if (
                    elapsed >= self.dynamic_shift_right_duration and
                    (target_reached or target_timed_out)
                ):
                    self.dynamic_stage = "DYN_HOLD_RIGHT"
                    self.dynamic_stage_started_at = now
                    self.dynamic_clear_start = None

            elif self.dynamic_stage == "DYN_HOLD_RIGHT":
                desired_offset = self.right_avoid_offset
                speed_cap = 11.0

                if now - self.dynamic_stage_started_at >= self.dynamic_right_hold:
                    self.dynamic_stage = "DYN_SHIFT_LEFT"
                    self.dynamic_stage_started_at = now
                    self.dynamic_clear_start = None
                    desired_offset = self.left_avoid_offset
                    speed_cap = 7.8

            elif self.dynamic_stage == "DYN_SHIFT_LEFT":
                desired_offset = self.left_avoid_offset
                speed_cap = 7.8
                elapsed = now - self.dynamic_stage_started_at
                target_reached = self.dynamic_shift_target_reached(
                    self.left_avoid_offset
                )
                target_timed_out = (
                    elapsed >=
                    self.dynamic_shift_left_duration +
                    self.dynamic_shift_timeout_extra
                )

                if (
                    elapsed >= self.dynamic_shift_left_duration and
                    (target_reached or target_timed_out)
                ):
                    self.dynamic_stage = "DYN_HOLD_LEFT"
                    self.dynamic_stage_started_at = now
                    self.dynamic_clear_start = None

            elif self.dynamic_stage == "DYN_HOLD_LEFT":
                desired_offset = self.left_avoid_offset
                speed_cap = 11.0

                if now - self.dynamic_stage_started_at >= self.dynamic_left_hold:
                    self.dynamic_stage = "DYN_RETURN_CENTER"
                    self.dynamic_stage_started_at = now
                    self.dynamic_clear_start = None
                    desired_offset = 0.0
                    speed_cap = 8.5

            elif self.dynamic_stage == "DYN_RETURN_CENTER":
                desired_offset = 0.0
                speed_cap = 8.5

                if abs(self.obstacle_offset) < 2.0:
                    self.dynamic_stage = "IDLE"
                    self.dynamic_cooldown_until = now + 2.0
                    speed_cap = None

        self.obstacle_target_offset = desired_offset
        self.obstacle_speed_cap = speed_cap

        if self.dynamic_stage != self.last_reported_dynamic_stage:
            self.get_logger().info(
                f"mission stage: {self.last_reported_dynamic_stage} -> {self.dynamic_stage}"
            )
            self.last_reported_dynamic_stage = self.dynamic_stage

        if self.dynamic_stage == "DYN_SHIFT_RIGHT":
            max_delta = 36.0
        elif self.dynamic_stage == "DYN_SHIFT_LEFT":
            max_delta = 24.0
        else:
            max_delta = 16.0 if desired_offset == 0.0 else 22.0
        delta = np.clip(
            desired_offset - self.obstacle_offset,
            -max_delta,
            max_delta
        )
        self.obstacle_offset += float(delta)

        if desired_offset == 0.0 and abs(self.obstacle_offset) < 1.0:
            self.obstacle_offset = 0.0

        self.publish_avoidance_debug(now)
        return speed_cap

    def is_avoidance_stage_active(self):
        return self.dynamic_stage in (
            "PED_AVOID",
            "LIDAR_AVOID_RIGHT",
            "LIDAR_AVOID_LEFT",
        )

    def lidar_points_for_planner(self, min_range=0.35, max_range=5.5):
        points, _ = self.scan_points_with_indices(
            min_range=min_range,
            max_range=max_range,
        )
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        roi = (
            (points[:, 0] > 0.1) &
            (points[:, 0] < max_range) &
            (np.abs(points[:, 1]) < 2.4)
        )
        return points[roi]

    def rollout_trajectory(self, angle_cmd, horizon=3.2, step=0.22):
        angle_cmd = float(np.clip(angle_cmd, -90.0, 90.0))
        steer_rad = np.deg2rad(angle_cmd / 100.0 * 30.0)
        wheel_base = 0.33
        curvature = math.tan(steer_rad) / wheel_base

        x = 0.0
        y = 0.0
        yaw = 0.0
        samples = []

        distance = 0.0
        while distance < horizon:
            x += step * math.cos(yaw)
            y += step * math.sin(yaw)
            yaw += curvature * step
            distance += step
            samples.append((x, y))

        return np.asarray(samples, dtype=np.float32)

    def trajectory_clearance(self, trajectory, points):
        if points.size == 0:
            return 5.5

        deltas = trajectory[:, None, :] - points[None, :, :]
        dists = np.linalg.norm(deltas, axis=2)
        return float(np.min(dists))

    def plan_avoidance_motion(self, fallback_angle, speed_cap):
        if not self.is_avoidance_stage_active():
            return None

        points = self.lidar_points_for_planner()

        # Convert image-space avoidance offset into scan-local lateral target.
        # Negative offset means right in the current lane controller, and scan y
        # positive is right, so the signs are intentionally inverted here.
        target_y = float(np.clip(-self.obstacle_target_offset / 160.0, -1.15, 1.15))

        if self.dynamic_stage == "DYN_RETURN_CENTER":
            target_y = 0.0

        candidate_angles = np.linspace(-85.0, 85.0, 23)
        best = None
        best_score = -1e9
        best_clearance = 0.0

        for candidate in candidate_angles:
            trajectory = self.rollout_trajectory(candidate)
            clearance = self.trajectory_clearance(trajectory, points)

            if clearance < 0.42:
                continue

            final_y = float(trajectory[-1, 1])
            lateral_error = abs(final_y - target_y)
            angle_change = abs(candidate - self.planner_last_angle)
            steering_effort = abs(candidate)

            score = (
                clearance * 3.2
                - lateral_error * 2.6
                - steering_effort * 0.012
                - angle_change * 0.010
            )

            if score > best_score:
                best_score = score
                best = candidate
                best_clearance = clearance

        if best is None:
            self.planner_last_angle = 0.0
            return 0.0, 0.0

        if best_clearance < 0.60:
            planned_speed = 4.0
        elif best_clearance < 0.85:
            planned_speed = 6.0
        else:
            planned_speed = 8.0

        if speed_cap is not None:
            planned_speed = min(planned_speed, speed_cap)

        # Blend with the lane controller a little so the maneuver does not snap.
        planned_angle = 0.82 * best + 0.18 * fallback_angle
        planned_angle = float(np.clip(planned_angle, -90.0, 90.0))
        self.planner_last_angle = planned_angle

        return planned_angle, planned_speed

    #=============================================
    # [수정] 변경된 차선 픽셀 배열(fit_x) 수신 콜백 함수
    #=============================================
    def lane_fit_x_callback(self, msg):
        # 배열 데이터가 비어있지 않고 차선 유효성이 정상일 때 제어 실행
        self.last_fit_x_time = time.monotonic()
        if msg.data:
            self.lane_valid_status = True
            fit_x = np.array(msg.data)
            self.last_fit_x = fit_x
            self.process_autonomous_driving(fit_x)
        else:
            self.last_fit_x = None

    #=============================================
    # 외부 차선 노드로부터 유효성을 받는 콜백 함수
    #=============================================
    def lane_valid_callback(self, msg):
        self.lane_valid_status = msg.data
        if not self.lane_valid_status and not self.cone_valid_status:
            self.current_state = "EMERGENCY"
            # 차선 소실 즉시 비상 주행 로직 가동
            self.process_autonomous_driving(None)

    def cone_valid_callback(self, msg):
        self.cone_valid_status = msg.data
        if not self.cone_valid_status and not self.lane_valid_status:
            self.current_state = "EMERGENCY"
            self.process_autonomous_driving(None)

    #=============================================
    # 카메라 토픽을 수신하는 콜백 함수
    #=============================================
    def cam_callback(self, data):
        # 수신한 메시지를 OpenCV 이미지로 변환하여 저장
        self.last_camera_time = time.monotonic()
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    
    #=============================================
    # 라이다 토픽을 수신하는 콜백 함수
    #=============================================
    def lidar_callback(self, msg):
        self.last_scan_time = time.monotonic()
        self.lidar_ranges = msg.ranges
        self.latest_scan = msg
      
    #=============================================
    # 모터제어 토픽을 발행하는 Publisher 함수
    #=============================================
    def drive(self, angle, speed):
        self.motor_msg.header.stamp = self.get_clock().now().to_msg()
        self.motor_msg.header.frame_id = "base_link"
        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)
        self.last_cmd_angle = float(angle)
        self.last_cmd_speed = float(speed)
        self.motor_pub.publish(self.motor_msg)

    ##################################################
    # STEERING CALCULATION
    ##################################################

    def calculate_angle(
        self,
        fit_x,
        image_width,
        image_height,
        look_ahead_ratio,
        Kp,
        offset=0
    ):

        image_center = image_width//2
        center_offset = offset
        target_center = image_center + center_offset

        look_ahead_index = int((image_height - 1) * look_ahead_ratio)
        look_ahead_index = int(np.clip(look_ahead_index, 0, len(fit_x) - 1))
        target_lane_x = fit_x[look_ahead_index]

        error = target_lane_x - target_center

        steering_deg = Kp*error

        steering_deg = np.clip(
            steering_deg,
            -20,
            20
        )

        angle_cmd = steering_deg * 5.0

        return angle_cmd

    def apply_lane_safe_avoidance_bias(self, base_angle):
        if abs(self.obstacle_offset) < 1.0:
            return float(base_angle)

        # Keep the original lane controller in charge. The obstacle offset only
        # adds a small steering bias so an avoidance false-positive cannot yank
        # the vehicle out of the lane.
        bias = float(np.clip(-self.obstacle_offset * 0.46, -52.0, 52.0))
        limited_angle = float(np.clip(base_angle + bias, -85.0, 85.0))
        return limited_angle
    
    def calculate_curvature(self, fit_x, image_height):
        """
        Sliding Window의 2차 다항식 결과(fit_x)로부터 곡률 반경(R)을 계산합니다.
        fit_x는 이미지의 모든 y축(0 ~ h-1)에 대응하는 x 좌표 배열입니다.
        """
        try:
            # np.linspace로 fit_x와 매칭되는 y 배열 생성
            plot_y = np.linspace(0, image_height - 1, len(fit_x))
            
            # fit_x와 plot_y를 이용해 2차 다항식 계수(A, B, C)를 역으로 추출
            poly_coeffs = np.polyfit(plot_y, fit_x, 2)
            A = poly_coeffs[0]
            B = poly_coeffs[1]

            if np.abs(A) < 1e-5:
                return 99999.0
            
            # 차량 직전 전방(이미지 맨 하단 y지점)에서의 곡률 반경 계산
            y_eval = image_height - 1
            
            # 공식 적용: R = [1 + (2Ay + B)^2]^1.5 / |2A|
            numerator = (1 + (2 * A * y_eval + B) ** 2) ** 1.5
            denominator = np.abs(2 * A)
            
            if denominator < 1e-6:  # 분모가 0이 되는 것 방지 (완벽한 직선)
                return 99999.0
                
            curvature_radius = numerator / denominator
            return curvature_radius
            
        except Exception as e:
            return 99999.0

    #=============================================
    # 메인 루프
    #=============================================
    def main_loop(self):
    
        self.get_logger().info("======================================")
        self.get_logger().info("  S T A R T    D R I V I N G ...      ")
        self.get_logger().info("======================================")

        # rclpy.spin을 통해 이벤트 드리븐 방식으로 동작하도록 무한 루프 제어권 이양
        rclpy.spin(self)

    # ==============================================================================
    # [구조 고침] 상태 판단 FSM -> 가변 파라미터 적용 통합 파이프라인
    # ==============================================================================
    # ==============================================================================
    # [개선 고침] 3단계 다단 FSM 상태 판단 및 가변 파라미터 제어 파이프라인
    # ==============================================================================
    def process_autonomous_driving(self, fit_x):
        try:
            # Cone driving owns the command only while its virtual lane is valid.
            # The obstacle FSM is left untouched and resumes when cone mode ends.
            if self.cone_valid_status and fit_x is not None:
                self.current_state = "CONE_DRIVING"
                final_angle = self.calculate_angle(
                    fit_x,
                    self.img_w,
                    self.img_h,
                    0.55,
                    0.12,
                    0.0,
                )
                self.prev_angle = final_angle
                self.drive(final_angle, 10.0)
                return

            # 1. EMERGENCY 상태 탈출 먼저 처리 (안전장치)
            if self.current_state == "EMERGENCY" and self.lane_valid_status and fit_x is not None:
                self.current_state = "STRAIGHT"

            # 2. 차선 소실 상태가 아니라면 '곡률 반경(R)' 기반으로 상태(FSM) 판단
            if self.current_state != "EMERGENCY" and fit_x is not None:
                if self.current_state == "CONE_DRIVING":
                    self.current_state = "STRAIGHT"
                R = self.calculate_curvature(fit_x, self.img_h)
                
                SHARP_CURVE_THRESHOLD = 500.0  # 이 값보다 작으면 무조건 '심한 곡선'
                STRAIGHT_THRESHOLD = 1000.0    # 이 값보다 크면 '직선'
                
                if R <= SHARP_CURVE_THRESHOLD:
                    self.straight_consecutive_count = 0
                    self.current_state = "SHARP_CURVE"
                elif SHARP_CURVE_THRESHOLD < R <= STRAIGHT_THRESHOLD:
                    self.straight_consecutive_count = 0
                    self.current_state = "SOFT_CURVE"
                else:
                    self.straight_consecutive_count += 1

                # 직선 진입 노이즈 필터링
                STRAIGHT_STREAK_REQUIRED = 8  
                if self.straight_consecutive_count >= STRAIGHT_STREAK_REQUIRED:
                    self.current_state = "STRAIGHT"
                    self.straight_consecutive_count = STRAIGHT_STREAK_REQUIRED

            # 3. 3단계 상태별 가변 파라미터 적용 및 제어값 산출
            final_angle = 0.0
            final_speed = 0.0
            obstacle_speed_cap = self.update_obstacle_control()

            if self.current_state == "STRAIGHT":
                # [직선] 멀리 보고(0.45), 매우 부드럽게 조향(0.015)하여 털림/지그재그 방지
                target_ratio = 0.45 
                target_Kp = 0.025  
                final_speed = 13.0 if self.obstacle_offset != 0 else 16.0  
                
            elif self.current_state == "SOFT_CURVE":
                # [완만한 곡선] 중간을 보고(0.6), 적당한 강도로 조향(0.12) 및 약간의 감속
                target_ratio = 0.6  
                target_Kp = 0.12    
                final_speed = 12.0
                
            elif self.current_state == "SHARP_CURVE":
                # [심한 곡선] 차량 바로 앞을 보고(0.8), 매우 강하게 회전(0.2) 및 확실한 감속
                target_ratio = 0.8 
                target_Kp = 0.2    
                final_speed = 8.5   
                
            elif self.current_state == "EMERGENCY":
                # [비상] 차선 소실 시 이전 조향각과 안전 서행 속도 유지
                target_ratio = 0.6
                target_Kp = 0.12
                final_angle = self.prev_angle
                final_speed = 7.0   
            else:
                target_ratio = 0.6
                target_Kp = 0.12
                final_speed = 5.0

            # 4. 조향각 계산 및 명령 발행
            if self.current_state != "EMERGENCY":
                lane_tracking_offset = self.dynamic_lane_tracking_offset()
                if self.is_dynamic_car_stage():
                    if self.is_dynamic_shift_stage():
                        target_ratio = 0.62
                        target_Kp = max(target_Kp, self.dynamic_shift_kp)
                    else:
                        target_ratio = 0.56
                        target_Kp = max(target_Kp, self.dynamic_hold_kp)
                    self.lane_target_error_px(lane_tracking_offset, target_ratio)

                # 차량 회피 HOLD/RETURN 구간에서는 노랑 점선을 차로 경계로 보고,
                # 오른쪽/왼쪽 차로 중심에 해당하는 offset을 그대로 차선 추종에 넣는다.
                base_angle = self.calculate_angle(
                    fit_x,
                    self.img_w,
                    self.img_h,
                    target_ratio,
                    target_Kp,
                    lane_tracking_offset,
                )
                planned_motion = self.plan_avoidance_motion(
                    base_angle,
                    obstacle_speed_cap,
                )
                if planned_motion is None:
                    if self.is_dynamic_lane_tracking_stage():
                        final_angle = float(base_angle)
                    else:
                        final_angle = self.apply_lane_safe_avoidance_bias(base_angle)
                else:
                    final_angle, planned_speed = planned_motion
                    final_speed = min(final_speed, planned_speed)
                self.prev_angle = final_angle

            if obstacle_speed_cap is not None:
                final_speed = min(final_speed, obstacle_speed_cap)

            if self.obstacle_angle_override is not None:
                final_angle = float(self.obstacle_angle_override)
                self.prev_angle = final_angle

            self.drive(final_angle, final_speed)

        except Exception as e:
            self.get_logger().warn(f"State Control Error: {e}")
            self.drive(self.prev_angle, 5.0)
            
#=============================================
# 메인 함수
#=============================================
def main(args=None):
      
    rclpy.init(args=args)
    node = TrackDriverNode()
	
    try:
        node.main_loop()
    except KeyboardInterrupt:

        pass
    finally:
        node.drive(angle=0, speed=0)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
