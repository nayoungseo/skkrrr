#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
#=============================================
# 본 프로그램은 자이트론에서 제작한 것입니다.
# 상업라이센스에 의해 제공되므로 무단배포 및 상업적 이용을 금합니다.
# 교육과 실습 용도로만 사용가능하며 외부유출은 금지됩니다.
#=============================================
import rclpy, time, cv2, os, math
import numpy as np
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

# ROS2 표준 메시지
from std_msgs.msg import Float32MultiArray, Bool, String

#=============================================
# ROS2 Node 클래스 정의
#=============================================
class TrackDriverNode(Node):

    def __init__(self):
        super().__init__('driver')
        self.get_logger().info('----- Xycar self-driving node started -----')
        
        # 상수값 및 초기값 설정
        self.image = None  
        self.motor_msg = XycarMotor()        
        self.lidar_ranges = None
        self.bridge = CvBridge()

        self.img_w = 640
        self.img_h = 480
        
        # ROS2 Publisher 설정
        self.motor_pub = self.create_publisher(XycarMotor, 'xycar_motor', 10)
        
        # ROS2 Subscriber 설정
        self.sub_front = self.create_subscription(
            Image, '/usb_cam/image_raw/front', self.cam_callback, qos_profile_sensor_data)

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)
        
        self.sub_fit_x = self.create_subscription(
            Float32MultiArray, '/vision/lane_fit_x', self.lane_fit_x_callback, 10)
            
        self.sub_lane_valid = self.create_subscription(
            Bool, '/vision/lane_valid', self.lane_valid_callback, 10)
            
        self.sub_cone_valid = self.create_subscription(
            Bool, '/vision/cone_valid', self.cone_valid_callback, 10)

        self.sub_obstacle = self.create_subscription(
            String, '/lidar/obstacle_status', self.obstacle_status_callback, 10)

        # --------------------------------------------------------
        # 상태 제어 변수 초기화
        # --------------------------------------------------------
        self.prev_angle = 0.0  
        self.current_state = "CONE_DRIVING" # 초기 시작 상태를 라바콘 모드로 명시적 설정
        
        self.lane_valid_status = False
        self.cone_valid_status = False

        self.obstacle_state = "NONE"
        self.obstacle_offset = 0  
        self.straight_consecutive_count = 0

        self.get_logger().info("Track Driver Node Initialized with Dedicated CONE_DRIVING mode")

    #=============================================
    # 라이다 노드로부터 장애물 위험 신호 수신
    #=============================================
    def obstacle_status_callback(self, msg):
        # 🎯 [추가] 라바콘 주행 중일 때는 장애물 신호를 아예 접수하지 않음
        if self.current_state == "CONE_DRIVING":
            self.obstacle_state = "NONE"
            self.obstacle_offset = 0
            return

        self.obstacle_state = msg.data
        OFFSET_PIXEL = 130  
        
        if self.obstacle_state == "LEFT":
            self.obstacle_offset = -OFFSET_PIXEL
        elif self.obstacle_state == "RIGHT":
            self.obstacle_offset = OFFSET_PIXEL
        else:
            self.obstacle_offset = 0

    #=============================================
    # 차선 및 가상 차선 픽셀 배열(fit_x) 수신 콜백 함수
    #=============================================
    def lane_fit_x_callback(self, msg):
        is_any_valid = self.lane_valid_status or self.cone_valid_status
        if msg.data and is_any_valid:
            fit_x = np.array(msg.data)
            self.process_autonomous_driving(fit_x)

    #=============================================
    # 카메라 노드 유효성 수신 콜백
    #=============================================
    def lane_valid_callback(self, msg):
        self.lane_valid_status = msg.data
        self.check_emergency_fallback()

    #=============================================
    # 라바콘 노드 유효성 수신 콜백
    #=============================================
    def cone_valid_callback(self, msg):
        self.cone_valid_status = msg.data
        self.check_emergency_fallback()

    def check_emergency_fallback(self):
        # 어떤 노드도 유효 차선을 주지 못할 때만 EMERGENCY 상태로 빠집니다.
        if not self.lane_valid_status and not self.cone_valid_status:
            if self.current_state != "EMERGENCY":
                self.current_state = "EMERGENCY"
                self.process_autonomous_driving(None)

    def cam_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    
    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges   
      
    def drive(self, angle, speed):
        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)
        self.motor_pub.publish(self.motor_msg)

    ##################################################
    # STEERING CALCULATION (조향 제어 연산)
    ##################################################
    def calculate_angle(self, fit_x, image_width, image_height, look_ahead_ratio, Kp, offset=0):
        image_center = image_width // 2
        center_offset = offset
        target_center = image_center + center_offset

        # 🎯 [인덱싱 수정] look_ahead_ratio가 커질수록 하단(근거리 차량 코앞)을 정방향 조준합니다.
        look_ahead_index = int((image_height - 1) * look_ahead_ratio)
        look_ahead_index = np.clip(look_ahead_index, 0, len(fit_x) - 1)
        
        target_lane_x = fit_x[look_ahead_index]
        error = target_lane_x - target_center

        steering_deg = Kp * error
        steering_deg = np.clip(steering_deg, -20.0, 20.0)

        angle_cmd = steering_deg * 5.0
        return angle_cmd
    
    def calculate_curvature(self, fit_x, image_height):
        try:
            plot_y = np.linspace(0, image_height - 1, len(fit_x))
            poly_coeffs = np.polyfit(plot_y, fit_x, 2)
            A = poly_coeffs[0]
            B = poly_coeffs[1]
            
            if np.abs(A) < 1e-5:
                return 99999.0

            y_eval = image_height - 1
            numerator = (1 + (2 * A * y_eval + B) ** 2) ** 1.5
            denominator = np.abs(2 * A)
            
            if denominator < 1e-6:
                return 99999.0
                
            curvature_radius = numerator / denominator
            return curvature_radius
        except Exception as e:
            return 99999.0

    def main_loop(self):
        self.get_logger().info("======================================")
        self.get_logger().info("  S T A R T    D R I V I N G ...      ")
        self.get_logger().info("======================================")
        rclpy.spin(self)

    # ==============================================================================
    # 🎯 [구조 개선] 전용 CONE_DRIVING 상태가 추가된 자율주행 주행 파이프라인
    # ==============================================================================
    def process_autonomous_driving(self, fit_x):
        try:
            # 1. EMERGENCY 상태 복구 조건 검사
            if self.current_state == "EMERGENCY" and fit_x is not None:
                if self.cone_valid_status:
                    self.current_state = "CONE_DRIVING"
                elif self.lane_valid_status:
                    self.current_state = "STRAIGHT"

            # 2. 유효 플래그에 따른 마스터 상태 판단 (라바콘 주행 모드 전환 우선권 제어)
            if self.current_state != "EMERGENCY" and fit_x is not None:
                
                if self.cone_valid_status:
                    # 🎯 라바콘 유효 신호가 살아있다면 복잡한 곡률 계산 FSM을 전면 스킵하고 모드 고정!
                    self.current_state = "CONE_DRIVING"
                    self.obstacle_offset = 0
                    
                elif self.lane_valid_status:
                    # 라바콘 모드가 끝나고 진짜 카메라 차선이 켜지면 원래의 곡률 기반 3단계 FSM 구동
                    if self.current_state == "CONE_DRIVING": 
                        self.current_state = "STRAIGHT" # 라바콘에서 차선모드로 변환 시 초기화
                    
                    R = self.calculate_curvature(fit_x, self.img_h)
                    SHARP_CURVE_THRESHOLD = 500.0  
                    STRAIGHT_THRESHOLD = 1000.0    
                    
                    if R <= SHARP_CURVE_THRESHOLD:
                        self.straight_consecutive_count = 0
                        self.current_state = "SHARP_CURVE"
                    elif SHARP_CURVE_THRESHOLD < R <= STRAIGHT_THRESHOLD:
                        self.straight_consecutive_count = 0
                        self.current_state = "SOFT_CURVE"
                    else:
                        self.straight_consecutive_count += 1

                    STRAIGHT_STREAK_REQUIRED = 8  
                    if self.straight_consecutive_count >= STRAIGHT_STREAK_REQUIRED:
                        self.current_state = "STRAIGHT"
                        self.straight_consecutive_count = STRAIGHT_STREAK_REQUIRED

            # 3. 각 주행 모드(FSM 상태)별 가변 파라미터 적용 및 명령 산출
            final_angle = 0.0
            final_speed = 0.0

            # 🎯 [신규 추가] 라바콘 전용 주행 제어 파라미터 영역
            if self.current_state == "CONE_DRIVING":
                # 차량 코앞 가이드라인 중심(0.80)을 조준하고 강한 계수(0.32)로 꺾어 민첩성 최적화
                target_ratio = 0.55 
                target_Kp = 0.12    
                final_speed = 10.0 
                
            # --------------------------------------------------------
            # 카메라 차선 주행 파라미터 영역 (기존 요구 조건 그대로 유지)
            # --------------------------------------------------------
            elif self.current_state == "STRAIGHT":
                target_ratio = 0.45 
                target_Kp = 0.025   
                final_speed = 13.0 if self.obstacle_offset != 0 else 16.0  
                
            elif self.current_state == "SOFT_CURVE":
                target_ratio = 0.60 
                target_Kp = 0.12    
                final_speed = 12.0
                
            elif self.current_state == "SHARP_CURVE":
                target_ratio = 0.80 
                target_Kp = 0.20    
                final_speed = 8.5   
                
            elif self.current_state == "EMERGENCY":
                target_ratio = 0.60
                target_Kp = 0.12
                final_angle = self.prev_angle
                final_speed = 7.0   
            else:
                target_ratio = 0.60
                target_Kp = 0.12
                final_speed = 5.0

            # 4. 최종 조향각 연산 및 모터 제어 명령 발행
            if self.current_state != "EMERGENCY":
                
                # 🎯 [수정 핵심] 현재 상태가 라바콘 주행 모드(CONE_DRIVING)라면
                # 별도의 장애물 오프셋을 0으로 강제 초기화하여 이중 회피를 방지합니다.
                current_offset = self.obstacle_offset
                if self.current_state == "CONE_DRIVING":
                    current_offset = 0  # 라바콘 자체를 장애물로 오인해 이중으로 꺾는 현상 원천 차단
                
                final_angle = self.calculate_angle(
                    fit_x, self.img_w, self.img_h, target_ratio, target_Kp, current_offset)
                self.prev_angle = final_angle

            self.drive(final_angle, final_speed)

        except Exception as e:
            self.get_logger().warn(f"State Control Error in FSM: {e}")
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