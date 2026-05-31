#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Bool, Float32
import math
import numpy as np

class LidarProcessorNode(Node):
    def __init__(self):
        super().__init__('lidar_processor_node')

        self.lidar_ranges = None
        self.lane_valid_status = False

        # --------------------------------------------------------
        # ROS2 Subscriber / Publisher 설정 (원본 구조 100% 유지)
        # --------------------------------------------------------
        self.sub_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.sub_lane_valid = self.create_subscription(Bool, '/vision/lane_valid', self.lane_valid_callback, 10)

        self.pub_obstacle = self.create_publisher(String, '/lidar/obstacle_status', 10)
        self.pub_cone_angle = self.create_publisher(Float32, '/lidar/cone_angle', 10)
        self.pub_cone_valid = self.create_publisher(Bool, '/lidar/cone_valid', 10)

        self.obstacle_msg = String()
        self.cone_angle_msg = Float32()
        self.cone_valid_msg = Bool()

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("✅ Lidar Processor Node Started (Multi-Zone Detection Active)")

    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges   

    def lane_valid_callback(self, msg):
        self.lane_valid_status = msg.data

    def timer_callback(self):
        if self.lidar_ranges is None or len(self.lidar_ranges) == 0:
            self.get_logger().warn("No LiDAR data yet...")
            return

        ranges = self.lidar_ranges
        self.process_lavacone_calculations(ranges)
        self.process_obstacle_calculations(ranges)

    def process_lavacone_calculations(self, ranges):
        # (라바콘 주행 함수는 기존 코드와 완전히 동일하므로 생략하지 않고 그대로 유지)
        total_points = len(ranges)
        mid = total_points // 2  
        cone_span = int(total_points * (45.0 / 360.0))
        left_start = mid + int(total_points * (15.0 / 360.0))
        left_end = left_start + cone_span
        right_end = mid - int(total_points * (15.0 / 360.0))
        right_start = right_end - cone_span

        left_cone_dist = [r for r in ranges[left_start:left_end] if math.isfinite(r) and 0.1 < r < 1.5]
        right_cone_dist = [r for r in ranges[right_start:right_end] if math.isfinite(r) and 0.1 < r < 1.5]

        avg_left = np.mean(left_cone_dist) if len(left_cone_dist) > 0 else 1.5
        avg_right = np.mean(right_cone_dist) if len(right_cone_dist) > 0 else 1.5

        cone_error = avg_left - avg_right  
        Kp_cone = 80.0 
        cone_angle = cone_error * Kp_cone
        cone_angle = float(np.clip(cone_angle, -20.0, 20.0))

        is_cone_valid = not self.lane_valid_status

        self.cone_angle_msg.data = cone_angle
        self.cone_valid_msg.data = is_cone_valid
        self.pub_cone_angle.publish(self.cone_angle_msg)
        self.pub_cone_valid.publish(self.cone_valid_msg)

    #=============================================
    # [연산 2] 차량 회피용 장애물 상태 계산 (멀티존 튜닝 버전)
    #=============================================
    def process_obstacle_calculations(self, ranges):
        total_points = len(ranges)
        
        # --------------------------------------------------------
        # ⚙️ [가변 파라미터 튜닝 구역 - 원거리/근거리 이중화]
        # 시뮬레이션 환경에 따라 거리(m)와 각도를 조율하세요.
        # --------------------------------------------------------
        # 최소 데드존 (범용)
        dist_min = 0.2
        
        # ZONE 1: 좁고 길게 보는 영역 (5도 ~ 25도)
        z1_angle_min = 5.0
        z1_angle_max = 25.0
        z1_dist_max  = 8   # 👈 5~25도는 길게 감시 (1.5미터)
        
        # ZONE 2: 넓고 짧게 보는 영역 (26도 ~ 90도)
        z2_angle_min = 26.0
        z2_angle_max = 90.0
        z2_dist_max  = 3   # 👈 26~90도는 짧게 감시 (0.6미터)
        
        trigger_count = 4    # 장애물 인정을 위한 최소 포인트 개수
        # --------------------------------------------------------

        # 각도 비율을 인덱스 카운트로 환산
        z1_narrow = int(total_points * (z1_angle_min / 360.0))
        z1_wide   = int(total_points * (z1_angle_max / 360.0))
        
        z2_narrow = int(total_points * (z2_angle_min / 360.0))
        z2_wide   = int(total_points * (z2_angle_max / 360.0))

        # --- 인덱스 슬라이싱 구역 분할 (정면 0도, 반시계+ 기준) ---
        # 1-1. 좌측 원거리 존 (+5 ~ +25)
        left_z1 = ranges[z1_narrow : z1_wide]
        # 1-2. 좌측 근거리 존 (+26 ~ +90)
        left_z2 = ranges[z2_narrow : z2_wide]
        
        # 2-1. 우측 원거리 존 (-25 ~ -5 => 335 ~ 355)
        right_z1 = ranges[total_points - z1_wide : total_points - z1_narrow]
        # 2-2. 우측 근거리 존 (-90 ~ -26 => 270 ~ 334)
        right_z2 = ranges[total_points - z2_wide : total_points - z2_narrow]

        # --- 각 구역별 고유 거리 임계값을 대입하여 카운트 ---
        left_z1_cnt = sum(1 for r in left_z1 if math.isfinite(r) and dist_min < r < z1_dist_max)
        left_z2_cnt = sum(1 for r in left_z2 if math.isfinite(r) and dist_min < r < z2_dist_max)
        
        right_z1_cnt = sum(1 for r in right_z1 if math.isfinite(r) and dist_min < r < z1_dist_max)
        right_z2_cnt = sum(1 for r in right_z2 if math.isfinite(r) and dist_min < r < z2_dist_max)

        # 원거리든 근거리든 하나라도 카운트 기준을 넘으면 해당 방향 장애물로 판단
        left_detected  = (left_z1_cnt >= trigger_count) or (left_z2_cnt >= trigger_count)
        right_detected = (right_z1_cnt >= trigger_count) or (right_z2_cnt >= trigger_count)

        # 🔀 [예외 처리] 좌우 모두 장애물이 감지된 경우!
        if left_detected and right_detected:
            # 유효한 거리 정보만 추출 (inf 제외)
            left_valid_v = [r for r in list(left_z1)+list(left_z2) if math.isfinite(r) and dist_min < r < 1.5]
            right_valid_v = [r for r in list(right_z1)+list(right_z2) if math.isfinite(r) and dist_min < r < 1.5]
            
            avg_left_dist = np.mean(left_valid_v) if len(left_valid_v) > 0 else 1.5
            avg_right_dist = np.mean(right_valid_v) if len(right_valid_v) > 0 else 1.5
            
            # 왼쪽 장애물이 더 가깝다 = 오른쪽에 공간이 더 많다 ➡️ RIGHT 상태 발행 (우회전 유도)
            if avg_left_dist < avg_right_dist:
                status_str = "LEFT" # (기존 매핑 방식 유지: 왼쪽에 차가 있으니 우회전하라는 뜻)
            
            # 오른쪽 장애물이 더 가깝다 = 왼쪽에 공간이 더 많다 ➡️ LEFT 상태 발행 (좌회전 유도)
            else:
                status_str = "RIGHT"

        # 기존 단일 감지 로직
        elif left_detected:
            status_str = "LEFT"
        elif right_detected:
            status_str = "RIGHT"
        else:
            status_str = "NONE"

        # 🖥️ 디버깅 전용 원라인 모니터링 로그 출력
        print(f"[라이다] 상태: {status_str:<5} | 좌측(원거{left_z1_cnt} / 근거{left_z2_cnt}) | 우측(원거{right_z1_cnt} / 근거{right_z2_cnt})", end='\r', flush=True)

        # 토픽 발행
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