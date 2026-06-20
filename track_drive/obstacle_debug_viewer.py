#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String


class ObstacleDebugViewer(Node):
    def __init__(self):
        super().__init__('obstacle_debug_viewer')

        self.bridge = CvBridge()
        self.image = None
        self.lidar_ranges = None
        self.obstacle_status = "NONE"
        self.driver_debug = {}
        self.last_driver_debug_time = 0.0
        self.started_at = time.monotonic()
        self.obstacle_detection_delay = 15.0

        self.pedestrian_seen_until = 0.0
        self.lidar_pedestrian_seen_until = 0.0

        self.sub_front = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.sub_scan = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.sub_obstacle = self.create_subscription(
            String,
            '/lidar/obstacle_status',
            self.obstacle_status_callback,
            10,
        )
        self.sub_driver_debug = self.create_subscription(
            String,
            '/track_drive/avoidance_debug',
            self.driver_debug_callback,
            10,
        )

        self.timer = self.create_timer(0.05, self.render)
        self.get_logger().info("Obstacle debug viewer started")

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def scan_callback(self, msg):
        self.lidar_ranges = msg.ranges

    def obstacle_status_callback(self, msg):
        if not self.obstacle_detection_enabled():
            self.obstacle_status = "NONE"
            return
        self.obstacle_status = msg.data.strip().upper()

    def driver_debug_callback(self, msg):
        try:
            self.driver_debug = json.loads(msg.data)
        except json.JSONDecodeError:
            self.driver_debug = {"stage": msg.data.strip()}
        self.last_driver_debug_time = time.monotonic()

    def obstacle_detection_enabled(self):
        return time.monotonic() - self.started_at >= self.obstacle_detection_delay

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
        stage = self.driver_debug.get("stage", "IDLE")
        left_bound_ratio = 0.29 if stage == "PED_STOP" else 0.34
        left_bound = width * (left_bound_ratio - 0.30 * t)
        right_bound = width * (0.68 + 0.31 * t)
        return left_bound <= x <= right_bound

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
        max_abs_y=1.75,
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
        groups = [
            group for group in np.split(raw_indices, split_at)
            if group.size >= min_points
        ]

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

    def detect_camera_pedestrians(self, frame):
        now = time.monotonic()
        h, w = frame.shape[:2]

        stage = self.driver_debug.get("stage", "IDLE")
        left_ratio = 0.06 if stage == "PED_STOP" else 0.14
        x1 = int(w * left_ratio)
        x2 = int(w * 0.92)
        y1 = int(h * 0.20)
        y2 = int(h * 0.92)
        roi = frame[y1:y2, x1:x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue_green = cv2.inRange(
            hsv,
            np.array([35, 45, 35]),
            np.array([135, 255, 255]),
        )
        dark = cv2.inRange(
            hsv,
            np.array([0, 0, 0]),
            np.array([180, 255, 80]),
        )
        mask = cv2.bitwise_or(blue_green, dark)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 15), np.uint8))
        skin = cv2.inRange(
            hsv,
            np.array([0, 35, 70]),
            np.array([28, 210, 255]),
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []
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
                    np.array([95, 255, 255]),
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

            candidate = {
                "bbox": (global_x, global_y, bw, bh),
                "center_x": center_x,
                "bottom_y": bottom_y,
                "area": area,
                "score": area * aspect + upper_skin * 4.0 + lower_dark * 0.6,
                "active": False,
            }
            candidates.append(candidate)

            if candidate["score"] > best_score:
                best_score = candidate["score"]
                best = candidate

        active = False
        if best is not None:
            close_enough = best["bottom_y"] > h * 0.45 and best["area"] > 230
            active = close_enough
            if active:
                self.pedestrian_seen_until = now + 0.35
                best["active"] = True

        active = active or now < self.pedestrian_seen_until
        return candidates, active

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
        source,
    ):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
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
                image_height,
            ):
                continue

            candidates.append({
                "bbox": (roi_x + x, roi_y + y, bw, bh),
                "center_x": center_x,
                "center_y": center_y,
                "area": area,
                "source": source,
            })

        return candidates

    def detect_camera_vehicles(self, frame):
        now = time.monotonic()
        h, w = frame.shape[:2]

        x1 = int(w * 0.10)
        x2 = int(w * 0.90)
        y1 = int(h * 0.18)
        y2 = int(h * 0.78)
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        yellow_car = cv2.inRange(
            hsv,
            np.array([18, 60, 70]),
            np.array([95, 255, 255]),
        )
        yellow_car = cv2.morphologyEx(
            yellow_car,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )
        yellow_car = cv2.morphologyEx(
            yellow_car,
            cv2.MORPH_CLOSE,
            np.ones((7, 5), np.uint8),
        )

        red_low = cv2.inRange(
            hsv,
            np.array([0, 70, 70]),
            np.array([12, 255, 255]),
        )
        red_high = cv2.inRange(
            hsv,
            np.array([165, 70, 70]),
            np.array([180, 255, 255]),
        )
        rear_lights = cv2.bitwise_or(red_low, red_high)
        rear_lights = cv2.dilate(
            rear_lights,
            np.ones((11, 19), np.uint8),
            iterations=1,
        )
        rear_lights = cv2.morphologyEx(
            rear_lights,
            cv2.MORPH_CLOSE,
            np.ones((5, 9), np.uint8),
        )

        dark_vehicle = cv2.inRange(
            hsv,
            np.array([0, 0, 15]),
            np.array([180, 110, 135]),
        )
        dark_vehicle = cv2.morphologyEx(
            dark_vehicle,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )
        dark_vehicle = cv2.morphologyEx(
            dark_vehicle,
            cv2.MORPH_CLOSE,
            np.ones((9, 7), np.uint8),
        )

        candidates = []
        candidates.extend(self.vehicle_like_candidates_from_mask(
            yellow_car, x1, y1, w, h, 180, 10000, 18, 12, 95, "yellow",
        ))
        candidates.extend(self.vehicle_like_candidates_from_mask(
            rear_lights, x1, y1, w, h, 80, 7000, 16, 8, 90, "red",
        ))
        candidates.extend(self.vehicle_like_candidates_from_mask(
            dark_vehicle, x1, y1, w, h, 220, 12000, 20, 12, 110, "dark",
        ))

        merged = []
        for cand in sorted(candidates, key=lambda item: item["area"], reverse=True):
            duplicate = False
            for other in merged:
                if (
                    abs(cand["center_x"] - other["center_x"]) < 75 and
                    abs(cand["center_y"] - other["center_y"]) < 40
                ):
                    duplicate = True
                    break
            if not duplicate:
                merged.append(cand)

        active = False
        if len(merged) >= 2:
            xs = [item["center_x"] for item in merged]
            if max(xs) - min(xs) >= w * 0.12:
                active = True

        return merged, active

    def classify_lidar_clusters(self):
        clusters = self.lidar_corridor_clusters(max_x=6.4, max_abs_y=1.45)

        vehicle_clusters = []
        left_vehicle_clusters = []
        pedestrian_clusters = []
        for cluster in clusters:
            extent = max(cluster["span_x"], cluster["span_y"])

            is_pedestrian_like = (
                cluster["count"] <= 16 and
                cluster["span_x"] <= 0.95 and
                cluster["span_y"] <= 0.85 and
                cluster["median_x"] <= 4.6 and
                abs(cluster["median_y"]) <= 0.88
            )
            if is_pedestrian_like:
                pedestrian_clusters.append(cluster)
                continue

            vehicle_like = False
            if cluster["count"] >= 7 and extent >= 0.75:
                vehicle_like = True
            elif cluster["count"] >= 12 and extent >= 0.45:
                vehicle_like = True
            elif (
                cluster["count"] >= 8 and
                cluster["span_y"] >= 0.55 and
                cluster["span_x"] <= 1.25 and
                cluster["median_x"] > 1.0
            ):
                vehicle_like = True

            if vehicle_like:
                vehicle_clusters.append(cluster)
                if (
                    -1.05 <= cluster["median_y"] <= -0.22 and
                    cluster["min_x"] < 6.4
                ):
                    left_vehicle_clusters.append(cluster)

        dynamic_pair_active = False
        for i, first in enumerate(vehicle_clusters):
            for second in vehicle_clusters[i + 1:]:
                lateral_spread = abs(first["median_y"] - second["median_y"])
                nearest_x = min(first["min_x"], second["min_x"])
                opposite_sides = (
                    (first["median_y"] < -0.20 and second["median_y"] > 0.20) or
                    (second["median_y"] < -0.20 and first["median_y"] > 0.20)
                )
                if lateral_spread >= 0.60 and nearest_x < 6.2 and opposite_sides:
                    dynamic_pair_active = True

        lidar_ped_active = self.update_lidar_pedestrian_state(
            pedestrian_clusters,
            dynamic_pair_active,
        )

        return {
            "clusters": clusters,
            "vehicle_clusters": vehicle_clusters,
            "left_vehicle_clusters": left_vehicle_clusters,
            "pedestrian_clusters": pedestrian_clusters,
            "dynamic_pair_active": dynamic_pair_active,
            "pedestrian_active": lidar_ped_active,
        }

    def update_lidar_pedestrian_state(self, pedestrian_clusters, dynamic_pair_active):
        now = time.monotonic()
        if dynamic_pair_active or not pedestrian_clusters:
            return now < self.lidar_pedestrian_seen_until

        best = min(
            pedestrian_clusters,
            key=lambda cluster: (
                cluster["median_x"],
                abs(cluster["median_y"]),
            ),
        )
        obs_x = best["median_x"]
        obs_y = best["median_y"]

        stage = self.driver_debug.get("stage", "IDLE")
        left_limit = -1.08 if stage == "PED_STOP" else -0.78
        right_limit = 0.78
        active = obs_x < 2.85 and left_limit <= obs_y <= right_limit
        if active:
            self.lidar_pedestrian_seen_until = now + 0.35

        return active or now < self.lidar_pedestrian_seen_until

    def draw_road_roi(self, frame):
        h, w = frame.shape[:2]
        pts = np.array([
            [int(w * 0.36), int(h * 0.22)],
            [int(w * 0.64), int(h * 0.22)],
            [int(w * 0.92), int(h * 0.82)],
            [int(w * 0.08), int(h * 0.82)],
        ], dtype=np.int32)
        cv2.polylines(frame, [pts], True, (180, 180, 180), 1, cv2.LINE_AA)

    def draw_camera_overlays(
        self,
        frame,
        ped_candidates,
        ped_active,
        vehicle_candidates,
        vehicle_active,
        lidar_info,
    ):
        self.draw_road_roi(frame)

        for candidate in ped_candidates:
            x, y, bw, bh = candidate["bbox"]
            color = (0, 0, 255) if candidate["active"] or ped_active else (255, 255, 0)
            label = "CAM PED" if candidate["active"] or ped_active else "ped cand"
            cv2.rectangle(
                frame,
                (int(x), int(y)),
                (int(x + bw), int(y + bh)),
                color,
                2,
            )
            cv2.putText(
                frame,
                label,
                (int(x), max(18, int(y) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

        for candidate in vehicle_candidates:
            x, y, bw, bh = candidate["bbox"]
            color = (0, 165, 255) if vehicle_active else (0, 220, 220)
            cv2.rectangle(
                frame,
                (int(x), int(y)),
                (int(x + bw), int(y + bh)),
                color,
                2,
            )
            cv2.putText(
                frame,
                f"CAM CAR {candidate['source']}",
                (int(x), max(18, int(y) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        self.draw_projected_lidar_clusters(frame, lidar_info)

        status_color = (0, 255, 0)
        if self.obstacle_status != "NONE":
            status_color = (0, 0, 255)

        lines = [
            f"/lidar/obstacle_status: {self.obstacle_status}",
            f"CAM ped={ped_active}  CAM car=disabled",
            (
                f"LIDAR ped={lidar_info['pedestrian_active']}  "
                f"LIDAR left_obj={len(lidar_info['left_vehicle_clusters'])}"
            ),
            f"PED trigger={ped_active or lidar_info['pedestrian_active']}",
            (
                f"clusters: ped_like={len(lidar_info['pedestrian_clusters'])}  "
                f"veh_like={len(lidar_info['vehicle_clusters'])}"
            ),
        ]

        driver_age = time.monotonic() - self.last_driver_debug_time
        if self.driver_debug and driver_age < 1.0:
            stage = self.driver_debug.get("stage", "UNKNOWN")
            elapsed = float(self.driver_debug.get("stage_elapsed", 0.0) or 0.0)
            hold = self.driver_debug.get("stage_hold")
            trigger = self.driver_debug.get("car_trigger_mode", "none")
            lidar_status = self.driver_debug.get("left_lidar_status", "none")
            lidar_distance = self.driver_debug.get("left_lidar_distance")
            closure = self.driver_debug.get("left_lidar_closure_rate")
            current_offset = self.driver_debug.get("current_offset")
            target_offset = self.driver_debug.get("target_offset")
            lane_error = self.driver_debug.get("lane_target_error_px")
            heading_error = self.driver_debug.get("lane_heading_error_px")
            ped_source = self.driver_debug.get("pedestrian_source", "none")

            if stage == "DYN_SHIFT_RIGHT":
                avoid_text = f"avoiding car SHIFT RIGHT {elapsed:.1f}/{float(hold or 1.5):.1f}s"
            elif stage == "DYN_HOLD_RIGHT":
                avoid_text = f"avoiding car HOLD RIGHT {elapsed:.1f}/{float(hold or 4.0):.1f}s"
            elif stage == "DYN_SHIFT_LEFT":
                avoid_text = f"avoiding car SHIFT LEFT {elapsed:.1f}/{float(hold or 2.5):.1f}s"
            elif stage == "DYN_HOLD_LEFT":
                avoid_text = f"avoiding car HOLD LEFT {elapsed:.1f}/{float(hold or 7.0):.1f}s"
            elif stage == "DYN_RETURN_CENTER":
                avoid_text = f"avoiding car RETURN {elapsed:.1f}s"
            elif stage == "PED_STOP":
                avoid_text = f"ped stop {elapsed:.1f}/{float(hold or 3.0):.1f}s"
            elif stage == "PED_BACKUP":
                avoid_text = f"ped backup {elapsed:.1f}/{float(hold or 0.8):.1f}s"
            else:
                avoid_text = f"avoiding car off stage={stage}"

            closure_text = "None" if closure is None else f"{float(closure):.2f}m/s"
            distance_text = "None" if lidar_distance is None else f"{float(lidar_distance):.2f}m"
            current_offset_text = "None" if current_offset is None else f"{float(current_offset):.0f}px"
            target_offset_text = "None" if target_offset is None else f"{float(target_offset):.0f}px"
            lane_error_text = "None" if lane_error is None else f"{float(lane_error):.0f}px"
            heading_error_text = (
                "None"
                if heading_error is None
                else f"{float(heading_error):.0f}px"
            )
            lines.extend([
                avoid_text,
                (
                    f"avoid offset cur={current_offset_text} "
                    f"target={target_offset_text} "
                    f"lane_err={lane_error_text} head_err={heading_error_text}"
                ),
                (
                    f"car_trigger={trigger} left_lidar={lidar_status} "
                    f"dist={distance_text} closure={closure_text}"
                ),
                f"ped_source={ped_source}",
            ])
        else:
            lines.append("avoiding car: no recent driver debug")

        for i, text in enumerate(lines):
            y = 24 + i * 22
            color = status_color if i == 0 else (255, 255, 255)
            if "avoiding car SHIFT RIGHT" in text or "avoiding car HOLD RIGHT" in text:
                color = (0, 165, 255)
            elif "avoiding car SHIFT LEFT" in text or "avoiding car HOLD LEFT" in text:
                color = (255, 220, 0)
            elif "avoiding car RETURN" in text:
                color = (180, 255, 180)
            elif "ped backup" in text:
                color = (255, 180, 255)
            elif "ped stop" in text:
                color = (120, 220, 255)
            cv2.putText(
                frame,
                text,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
                cv2.LINE_AA,
            )

    def draw_projected_lidar_clusters(self, frame, lidar_info):
        h, w = frame.shape[:2]

        for cluster in lidar_info["clusters"]:
            x = max(cluster["median_x"], 0.45)
            y = cluster["median_y"]
            image_x = int(w * 0.5 + (y / x) * w * 0.55)
            image_y = int(h * 0.80 - np.clip(x / 6.8, 0.0, 1.0) * h * 0.42)
            if image_x < 0 or image_x >= w or image_y < 0 or image_y >= h:
                continue

            color = (180, 180, 180)
            label = "L"
            if cluster in lidar_info["vehicle_clusters"]:
                color = (0, 165, 255)
                label = "L-CAR"
            if cluster in lidar_info["left_vehicle_clusters"]:
                color = (0, 255, 255)
                label = "L-LEFT"
            if cluster in lidar_info["pedestrian_clusters"]:
                color = (255, 0, 255)
                label = "L-PED"

            cv2.circle(frame, (image_x, image_y), 7, color, -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                label,
                (image_x + 8, image_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    def make_lidar_panel(self, height, lidar_info):
        width = 320
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (25, 25, 25)

        cv2.putText(
            panel,
            "LiDAR corridor view",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        origin_x = width // 2
        origin_y = height - 38
        scale = min(38.0, (height - 80) / 7.0)

        def to_px(point_x, point_y):
            return (
                int(origin_x + point_y * scale),
                int(origin_y - point_x * scale),
            )

        points, _ = self.scan_points_with_indices(max_range=7.2)
        if points.size > 0:
            roi = (
                (points[:, 0] >= 0.0) &
                (points[:, 0] <= 7.0) &
                (np.abs(points[:, 1]) <= 3.0)
            )
            for point_x, point_y in points[roi][::2]:
                cv2.circle(panel, to_px(point_x, point_y), 1, (90, 90, 90), -1)

        left_top = to_px(6.8, -1.75)
        right_bottom = to_px(0.6, 1.75)
        cv2.rectangle(panel, left_top, right_bottom, (80, 80, 80), 1)
        cv2.line(panel, to_px(0.0, 0.0), to_px(6.8, 0.0), (90, 90, 90), 1)
        cv2.circle(panel, to_px(0.0, 0.0), 5, (0, 0, 255), -1)

        for cluster in lidar_info["clusters"]:
            color = (120, 120, 120)
            if cluster in lidar_info["vehicle_clusters"]:
                color = (0, 165, 255)
            if cluster in lidar_info["left_vehicle_clusters"]:
                color = (0, 255, 255)
            if cluster in lidar_info["pedestrian_clusters"]:
                color = (255, 0, 255)

            px = to_px(cluster["median_x"], cluster["median_y"])
            cv2.circle(panel, px, 6, color, -1, cv2.LINE_AA)
            cv2.putText(
                panel,
                f"{cluster['median_x']:.1f},{cluster['median_y']:+.1f}",
                (px[0] + 8, px[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                color,
                1,
                cv2.LINE_AA,
            )

        legend = [
            "gray: raw scan",
            "magenta: ped-like",
            "orange: vehicle-like",
            "cyan: left target",
            "y is right, x is forward",
            "q/esc: close",
        ]
        for i, text in enumerate(legend):
            cv2.putText(
                panel,
                text,
                (12, height - 118 + i * 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        return panel

    def render(self):
        if self.image is None:
            return

        frame = self.image.copy()
        if not self.obstacle_detection_enabled():
            left = self.obstacle_detection_delay - (time.monotonic() - self.started_at)
            ped_candidates, ped_active = [], False
            vehicle_candidates, vehicle_active = [], False
            lidar_info = {
                "clusters": [],
                "vehicle_clusters": [],
                "left_vehicle_clusters": [],
                "pedestrian_clusters": [],
                "dynamic_pair_active": False,
                "pedestrian_active": False,
            }
            cv2.putText(
                frame,
                f"OBSTACLE DETECTION OFF {max(0.0, left):.1f}s",
                (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.82,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            ped_candidates, ped_active = self.detect_camera_pedestrians(frame)
            vehicle_candidates, vehicle_active = [], False
            lidar_info = self.classify_lidar_clusters()

        self.draw_camera_overlays(
            frame,
            ped_candidates,
            ped_active,
            vehicle_candidates,
            vehicle_active,
            lidar_info,
        )
        panel = self.make_lidar_panel(frame.shape[0], lidar_info)
        debug_frame = np.hstack((frame, panel))

        cv2.imshow("track_drive obstacle debug", debug_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDebugViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
