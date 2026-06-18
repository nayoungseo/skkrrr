#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Bool
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import math
import cv2

class ConeDetectionNode(Node):
    def __init__(self):
        super().__init__('cone_detection')
        self.get_logger().info("===== Cone Detection Node Started =====")

        self.IMG_W = 640
        self.IMG_H = 480
        self.DEBUG = True

        self.lane_success_count = 0
        self.lane_active = False  

        # ⚙️ 범위를 늘려서 멀리 있는 왼쪽 콘까지 다 잡도록 설정 (MAX_Y: 7.5m -> 10.0m)
        self.MIN_X = -4.0
        self.MAX_X = 4.0
        self.MIN_Y = -3.5
        self.MAX_Y = 10.5

        # --------------------------------------------------------
        # SUBSCRIBE & PUBLISH
        # --------------------------------------------------------
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)
        
        self.sub_lane_valid = self.create_subscription(
            Bool, '/vision/lane_valid', self.lane_valid_callback, 10)

        self.fit_x_pub = self.create_publisher(Float32MultiArray, '/vision/lane_fit_x', 10)
        self.valid_pub = self.create_publisher(Bool, '/vision/cone_valid', 10)

        # 🎯 [추가] 직전 프레임의 피팅 계수를 기억할 변수 (초기값은 직진: 기울기 0, 절편 320)
        self.last_poly = np.array([0.0, 320.0])

    def lane_valid_callback(self, msg):
        if self.lane_active:
            return
        if msg.data is True:
            self.lane_success_count += 1
            if self.lane_success_count >= 8:
                self.get_logger().info("=====> Lane Detected 20+ consecutive frames! Switching to Lane Driving Mode.")
                self.lane_active = True
                if self.DEBUG:
                    cv2.destroyAllWindows()
        else:
            self.lane_success_count = 0

    def lidar_callback(self, msg):
        if self.lane_active:
            return

        ranges = msg.ranges
        actual_left_side_pixels = []  # 화면의 진짜 왼쪽에 찍히는 점들만 모을 배열

        # 1. 라이다 센서 데이터를 물리 X, Y 좌표(미터)로 변환
        for i, dist in enumerate(ranges):
            if not math.isfinite(dist) or dist < 1 or dist > 12.0:
                continue
            
            angle = np.deg2rad(i - 90)
            lx = -dist * np.cos(angle)
            ly = -dist * np.sin(angle)

            # 2. 전방 관심 영역(ROI) 필터링
            if self.MIN_X <= lx <= self.MAX_X and self.MIN_Y <= ly <= self.MAX_Y:
                
                # 3. 물리 좌표(미터) -> 이미지 픽셀 좌표 (640x480) 매핑 
                px = int((lx - self.MIN_X) / (self.MAX_X - self.MIN_X) * (self.IMG_W - 1))
                py = int((1.0 - (ly - self.MIN_Y) / (self.MAX_Y - self.MIN_Y)) * (self.IMG_H - 1))
                
                px = np.clip(px, 0, self.IMG_W - 1)
                py = np.clip(py, 0, self.IMG_H - 1)

                # 4. 이미지 화면 기준 "왼쪽 반토막(px < 320)"에 찍히는 진짜 왼쪽 콘 데이터만 수집
                if px < 320:  
                    if py < 120:
                        continue

                    if -0.4 < lx < 0.4:
                        continue
                    
                    # actual_left_side_pixels.append([px, py])

                    # 🎯 [수정] 피팅용 이미지 좌표(px, py)와 함께 '물리 Y거리(ly)'도 함께 저장합니다.
                    actual_left_side_pixels.append([px, py, ly])

        # 5. 수집된 진짜 왼쪽 콘들만 가지고 피팅 프로세스 진행
        self.process_cone_lanes(actual_left_side_pixels)

    def process_cone_lanes(self, left_pts):
        plot_y = np.linspace(0, self.IMG_H - 1, self.IMG_H)
        
        if self.DEBUG:
            debug_img = np.zeros((self.IMG_H, self.IMG_W, 3), dtype=np.uint8)
            # 차량 중심 위치 (320, 360) 및 전방 지시선
            cv2.rectangle(debug_img, (315, 355), (325, 365), (0, 0, 255), -1)
            cv2.line(debug_img, (320, 360), (320, 200), (0, 0, 255), 1)

        final_fit_x = None
        valid = False

        # 🎯 [핵심 검증 조건] 콘이 실질적으로 2개 이상 배치되어 있는지 확인
        is_enough_cones = False
        if len(left_pts) >= 2:
            pts_array = np.array(left_pts)
            ly_values = pts_array[:, 2] # 모든 점의 물리 Y 좌표들
            
            # 💡 점들이 퍼져있는 전방 거리의 최대-최소 차이를 구합니다.
            y_spread = np.max(ly_values) - np.min(ly_values)
            
            # 거리가 0.6m 이상 벌어져 있다면, 최소 2개 이상의 독립된 콘이 존재한다고 판정합니다.
            if y_spread >= 1:
                is_enough_cones = True

        # --------------------------------------------------------
        # 오직 선별된 왼쪽 콘들만 가지고 1차 직선 피팅 후 평행이동
        # --------------------------------------------------------
        # if len(left_pts) >= 2:
        if is_enough_cones:
            pts = np.array(left_pts)
            pixel_y = pts[:, 1]  # 이미지 상의 y 좌표 그대로 사용
            
            # 1차 선형 피팅 (Y좌표 입력 -> X좌표 출력)
            poly = np.polyfit(pixel_y, pts[:, 0], 1)
            
            # 🎯 [핵심] 성공한 피팅 값을 클래스 변수에 저장하여 기억합니다.
            self.last_poly = poly

            # 정방향 plot_y를 그대로 대입하여 차선 노드와 완전히 같은 인덱스 규칙을 만듭니다.
            # 인덱스 0 = y가 0일 때(원거리) / 인덱스 479 = y가 479일 때(근거리)
            left_fit_x = poly[0] * plot_y + poly[1]

            if self.DEBUG:
                for p in left_pts:
                    cv2.circle(debug_img, (int(p[0]), int(p[1])), 4, (0, 255, 255), -1)
                
                cone_fit_x_line = np.clip(left_fit_x, 0, self.IMG_W - 1)
                pts_cone_line = np.vstack((cone_fit_x_line, plot_y)).astype(np.int32).T
                cv2.polylines(debug_img, [pts_cone_line], isClosed=False, color=(255, 255, 0), thickness=2)
            
            # --------------------------------------------------------
            # 🎯 [수정] 동적 오프셋 고정 로직도 이미지 정방향에 맞춤
            # --------------------------------------------------------
            # 이미지 좌표계 기준 차량 코앞 높이인 y = 360 지점을 타겟으로 잡습니다.
            target_pixel_y = 360.0
            
            # 피팅된 라바콘 직선이 실제 이미지 y = 360 지점에서 가지는 x 좌표를 구합니다.
            current_x_at_360 = poly[0] * target_pixel_y + poly[1]
            
            # 이 x 좌표가 차량 중심축인 320이 되도록 오프셋을 계산합니다.
            LANE_OFFSET_PIXELS = 320.0 - current_x_at_360
            
            # 구한 오프셋을 더하면 이제 '360번 인덱스(=y가 360인 지점)'의 값이 정확히 320.0으로 고정됩니다.
            final_fit_x = left_fit_x + LANE_OFFSET_PIXELS  
            valid = True

        # --------------------------------------------------------
        # 2. 예외 상황 🎯: 왼쪽 콘이 1개 이하여서 직전 plot을 유지해야 할 때
        # --------------------------------------------------------
        else:
            # 🎯 [핵심] 새로 계산하지 않고, 가장 최근에 성공했던 self.last_poly를 그대로 재활용합니다.
            left_fit_x = self.last_poly[0] * plot_y + self.last_poly[1]
            
            # 저장되어 있던 계수(기울기, 절편) 기반으로 똑같이 차량 중심 오프셋을 다시 맞춰줍니다.
            target_pixel_y = 360.0
            current_x_at_360 = self.last_poly[0] * target_pixel_y + self.last_poly[1]
            LANE_OFFSET_PIXELS = 320.0 - current_x_at_360
            
            final_fit_x = left_fit_x + LANE_OFFSET_PIXELS
            valid = True # 제어 루프가 끊기지 않도록 True 유지

            if self.DEBUG:
                for p in left_pts:
                    cv2.circle(debug_img, (int(p[0]), int(p[1])), 4, (0, 255, 255), -1)
                cv2.putText(debug_img, "CONE LACK (<2) -> KEEP LAST PLOT MODE", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # --------------------------------------------------------
        # 최종 가상 차선 발행 및 디버깅 선 그리기
        # --------------------------------------------------------
        if valid and final_fit_x is not None:
            final_fit_x = np.clip(final_fit_x, 0, self.IMG_W - 1)
            
            fit_x_msg = Float32MultiArray()
            fit_x_msg.data = final_fit_x.tolist()
            self.fit_x_pub.publish(fit_x_msg)

            if self.DEBUG:
                pts_center = np.vstack((final_fit_x, plot_y)).astype(np.int32).T
                # 최종 제어용 가이드라인은 초록색 직선으로 표시
                cv2.polylines(debug_img, [pts_center], isClosed=False, color=(0, 255, 0), thickness=2)
                cv2.putText(debug_img, "CRISP LEFT-CONE MODE (CYAN=CONE LINE, GREEN=DRIVE LINE)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        valid_msg = Bool()
        valid_msg.data = valid
        self.valid_pub.publish(valid_msg)

        if self.DEBUG:
            cv2.imshow("CONE DETECTION DEBUG", debug_img)
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