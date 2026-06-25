#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String
from cv_bridge import CvBridge


class MissionVisionNode(Node):
    def __init__(self):
        super().__init__('mission_vision')

        self.bridge = CvBridge()

        # 어린이 보호구역 상태
        self.school_zone = False
        self.last_event_time = 0.0

        # 디버깅용 score
        self.last_school_score = 0.0
        self.last_release_score = 0.0

        # 튜닝 파라미터
        self.SCHOOL_TH = 0.45
        self.RELEASE_TH = 0.45
        self.MARGIN = 0.04
        self.LOCKOUT_SEC = 2.5

        # 어린이 보호구역 roi
        # frame[y1:y2, x1:x2]
        self.ROI_Y1 = 220
        self.ROI_Y2 = 480
        self.ROI_X1 = 40
        self.ROI_X2 = 600

        #신호등roi
        self.TL_Y1 = 50
        self.TL_Y2 = 180
        self.TL_X1 = 160
        self.TL_X2 = 500

        # [수정/추가] 신호등 인식을 위한 HSV 및 허프 변환 파라미터 정의
        self.SATURATION_TH = 70
        self.HUE_THRESHOLDS = {
            "RED":    ([0, 10], [170, 180]), # (낮은 영역, 높은 영역)
            "YELLOW": (15, 35),
            "GREEN":  (45, 95)
        }

        self.school_templates = []
        self.release_templates = []
        self.load_templates()

        self.sub_cam = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.pub_school_zone = self.create_publisher(
            Bool,
            '/mission/school_zone',
            10
        )

        # score 확인용. 없어도 주행에는 문제 없음.
        self.pub_school_score = self.create_publisher(
            Float32,
            '/mission/school_score',
            10
        )

        self.pub_release_score = self.create_publisher(
            Float32,
            '/mission/release_score',
            10
        )

        self.get_logger().info('===== mission_vision node started =====')

        #신호등
        self.traffic_light = "UNKNOWN"

        self.pub_traffic_light = self.create_publisher(
            String,
            '/mission/traffic_light',
            10
        )

    def load_templates(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(base_dir, 'templates')

        for i in range(1, 4):
            path = os.path.join(template_dir, f'school_zone_{i}.png')
            img = cv2.imread(path, cv2.IMREAD_COLOR)

            if img is None:
                self.get_logger().warn(f'cannot load template: {path}')
                continue

            processed = self.preprocess_yellow(img)
            self.school_templates.append(processed)

        for i in range(1, 4):
            path = os.path.join(template_dir, f'school_release_{i}.png')
            img = cv2.imread(path, cv2.IMREAD_COLOR)

            if img is None:
                self.get_logger().warn(f'cannot load template: {path}')
                continue

            processed = self.preprocess_yellow(img)
            self.release_templates.append(processed)

        self.get_logger().info(
            f'templates loaded: school={len(self.school_templates)}, release={len(self.release_templates)}'
        )

    def preprocess_yellow(self, bgr_img):
        """
        노란 바닥 글씨만 남기기 위한 전처리.
        template과 실시간 ROI에 동일하게 적용한다.
        """
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            np.array([15, 60, 60]),
            np.array([45, 255, 255])
        )

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return
        
        # [수정] 수정된 Hough Circle 기반의 신호등 판별 함수 호출
        light = self.detect_start_traffic_light(frame)
        self.traffic_light = light

        school_score, release_score = self.detect_school_zone(frame)

        self.last_school_score = school_score
        self.last_release_score = release_score

        self.update_school_zone(school_score, release_score)
        self.publish_result()

        self.show_debug(frame, school_score, release_score)

    def detect_school_zone(self, frame):
        roi = frame[self.ROI_Y1:self.ROI_Y2, self.ROI_X1:self.ROI_X2]
        roi_mask = self.preprocess_yellow(roi)

        school_score = self.best_template_score(roi_mask, self.school_templates)
        release_score = self.best_template_score(roi_mask, self.release_templates)

        return school_score, release_score

    def best_template_score(self, roi_mask, templates):
        if len(templates) == 0:
            return 0.0

        best = 0.0

        for template in templates:
            score = self.multi_scale_match(roi_mask, template)
            if score > best:
                best = score

        return best

    def multi_scale_match(self, roi_mask, template):
        best_score = 0.0

        scales = [0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50]

        roi_h, roi_w = roi_mask.shape[:2]
        th, tw = template.shape[:2]

        for scale in scales:
            new_w = int(tw * scale)
            new_h = int(th * scale)

            if new_w < 5 or new_h < 5:
                continue

            if new_w >= roi_w or new_h >= roi_h:
                continue

            resized = cv2.resize(template, (new_w, new_h))

            result = cv2.matchTemplate(
                roi_mask,
                resized,
                cv2.TM_CCOEFF_NORMED
            )

            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val

        return best_score

    def update_school_zone(self, school_score, release_score):
        now = time.time()

        if now - self.last_event_time < self.LOCKOUT_SEC:
            return

        if school_score > self.SCHOOL_TH and school_score > release_score + self.MARGIN:
            if not self.school_zone:
                self.school_zone = True
                self.last_event_time = now
                self.get_logger().info(
                    f'SCHOOL ZONE ON | school={school_score:.3f}, release={release_score:.3f}'
                )

        elif release_score > self.RELEASE_TH and release_score > school_score + self.MARGIN:
            if self.school_zone:
                self.school_zone = False
                self.last_event_time = now
                self.get_logger().info(
                    f'SCHOOL ZONE OFF | school={school_score:.3f}, release={release_score:.3f}'
                )

    def publish_result(self):
        msg = Bool()
        msg.data = self.school_zone
        self.pub_school_zone.publish(msg)

        s_msg = Float32()
        s_msg.data = float(self.last_school_score)
        self.pub_school_score.publish(s_msg)

        r_msg = Float32()
        r_msg.data = float(self.last_release_score)
        self.pub_release_score.publish(r_msg)

        light_msg = String()
        light_msg.data = self.traffic_light
        self.pub_traffic_light.publish(light_msg)

    def show_debug(self, frame, school_score, release_score):
        debug = frame.copy()

        cv2.rectangle(
            debug,
            (self.TL_X1, self.TL_Y1),
            (self.TL_X2, self.TL_Y2),
            (255, 0, 255),
            2
        )

        cv2.putText(
            debug,
            f'LIGHT={self.traffic_light}',
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2
        )

        cv2.rectangle(
            debug,
            (self.ROI_X1, self.ROI_Y1),
            (self.ROI_X2, self.ROI_Y2),
            (0, 255, 0),
            2
        )

        text1 = f'SCHOOL_ZONE={self.school_zone}'
        text2 = f'S={school_score:.3f} R={release_score:.3f}'

        cv2.putText(debug, text1, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(debug, text2, (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('mission_vision_debug', debug)
        cv2.waitKey(1)

    # =========================================================================
    # [수정 및 이식 영역] 예전 코드를 바탕으로 한 Hough Circle 기반의 신호등 인식 파트
    # =========================================================================
    
    def color_filtering(self, roi_hsv, color_str):
        """ 지정된 신호등 ROI 내에서 특정 색상의 마스크 이미지를 반환합니다. """
        h, s, v = cv2.split(roi_hsv)
        s_cond = s > self.SATURATION_TH

        if color_str == "RED":
            th_low, th_high = self.HUE_THRESHOLDS["RED"]
            h_cond = (h < th_low[1]) | (h > th_high[0])
        else:
            th = self.HUE_THRESHOLDS[color_str]
            h_cond = (h > th[0]) & (h < th[1])

        # 조건 만족 못하는 픽셀은 명도(v)를 0으로 처리
        v_filtered = np.where(h_cond & s_cond, v, 0).astype(np.uint8)
        
        hsv_filtered = cv2.merge([h, s, v_filtered])
        bgr_filtered = cv2.cvtColor(hsv_filtered, cv2.COLOR_HSV2BGR)
        return cv2.cvtColor(bgr_filtered, cv2.COLOR_BGR2GRAY)

    def detect_start_traffic_light(self, frame):
        """
        [수정 완] 출발선 3구 신호등의 현재 색상을 Hough Circles와 Color Filtering을 엮어 판단한다.
        반환값: "RED", "YELLOW", "GREEN", "UNKNOWN" 중 하나
        """
        # 1. 신호등 ROI 잘라내기 및 HSV 변환
        roi = frame[self.TL_Y1:self.TL_Y2, self.TL_X1:self.TL_X2]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 2. 각 색상별 원 검출 시도
        for color in ["RED", "YELLOW", "GREEN"]:
            gray_filtered = self.color_filtering(roi_hsv, color)
            
            # 허프 변환 원 검출 실행 (실제 환경에 맞게 마이너 튜닝이 필요할 수 있습니다)
            circles = cv2.HoughCircles(
                gray_filtered, 
                cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=40,
                param1=150, 
                param2=15,  # 원 판단 감도 (낮을수록 원을 더 잘 잡으나 노이즈 위험)
                minRadius=20, 
                maxRadius=50
            )

            # 원이 하나라도 검출되면 해당 색상으로 즉시 판정
            if circles is not None:
                return color

        # 아무 색상의 원도 검출되지 않았을 경우
        return "UNKNOWN"

def main(args=None):
    rclpy.init(args=args)
    node = MissionVisionNode()

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