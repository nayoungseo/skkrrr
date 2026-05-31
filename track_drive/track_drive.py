#!/usr/bin/env python3
# -*- coding: utf-8 -*- 1
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
from rclpy.duration import Duration
from cv_bridge import CvBridge

# [추가] 외부 차선 인식 노드의 토픽을 받기 위한 ROS2 표준 메시지
from std_msgs.msg import Float32MultiArray, Bool
# [추가] 외부 차선 및 라이다 노드의 토픽을 받기 위한 ROS2 표준 메시지
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
        self.bridge = CvBridge()

        # 차량 내부 이미지 크기 기본 규격 정의 (Sliding window 이미지 크기 기준)
        self.img_w = 640
        self.img_h = 480
        
        # ROS2 Publisher & Subscriber 설정
        self.motor_pub = self.create_publisher(XycarMotor,'xycar_motor',10)
        
        self.sub_front = self.create_subscription(
            Image, '/usb_cam/image_raw/front', self.cam_callback, qos_profile_sensor_data)

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)
		
        # [수정] /vision/lane_fit_x 토픽을 Float32MultiArray 데이터형으로 구독하도록 변경
        self.sub_fit_x = self.create_subscription(
            Float32MultiArray, '/vision/lane_fit_x', self.lane_fit_x_callback, 10)
            
        self.sub_valid = self.create_subscription(
            Bool, '/vision/lane_valid', self.lane_valid_callback, 10)

        # --------------------------------------------------------
        # [신규 추가] 라이다 장애물 상태 토픽 구독 설정
        # --------------------------------------------------------
        self.sub_obstacle = self.create_subscription(
            String, '/lidar/obstacle_status', self.obstacle_status_callback, 10)

        # --------------------------------------------------------
        # [신규 추가] 상태 제어 변수 및 모듈별 퍼블리셔 초기화
        # --------------------------------------------------------
        self.prev_angle = 0.0  # 차선 소실 시 유지할 이전 조향각 백업 변수
        self.current_state = "STRAIGHT" # 초기 차량 상태 (STRAIGHT / CURVE / EMERGENCY)
        self.lane_valid_status = True

        # [신규 변수] 실시간 라이다 회피 상태 및 변이 오프셋 저장 변수
        self.obstacle_state = "NONE"
        self.obstacle_offset = 0  

        self.straight_consecutive_count = 0

        self.get_logger().info("Track Driver Node Initialized")

    #=============================================
    # [신규 수신 콜백] 라이다 노드로부터 장애물 위험 신호 수신
    #=============================================
    def obstacle_status_callback(self, msg):
        self.obstacle_state = msg.data
        
        # ⚙️ [오프셋 픽셀 튜닝 포인트] 
        # 시뮬레이터 차량이 회피 기동을 너무 소심하게 하거나 과하게 할 때 이 값을 조정하세요.
        OFFSET_PIXEL = 130  
        
        if self.obstacle_state == "LEFT":
            # 왼쪽에 장애물 출현 -> 타겟 중심을 우측(+)으로 밀어 우회전 유도
            self.obstacle_offset = -OFFSET_PIXEL
        elif self.obstacle_state == "RIGHT":
            # 오른쪽에 장애물 출현 -> 타겟 중심을 좌측(-)으로 당겨 좌회전 유도
            self.obstacle_offset = OFFSET_PIXEL
        else:
            # 장애물 없음 -> 오프셋 초기화 (정중앙 주행)
            self.obstacle_offset = 0

    #=============================================
    # [수정] 변경된 차선 픽셀 배열(fit_x) 수신 콜백 함수
    #=============================================
    def lane_fit_x_callback(self, msg):
        # 배열 데이터가 비어있지 않고 차선 유효성이 정상일 때 제어 실행
        if msg.data and self.lane_valid_status:
            fit_x = np.array(msg.data)
            self.process_autonomous_driving(fit_x)

    #=============================================
    # 외부 차선 노드로부터 유효성을 받는 콜백 함수
    #=============================================
    def lane_valid_callback(self, msg):
        self.lane_valid_status = msg.data
        if not self.lane_valid_status:
            self.current_state = "EMERGENCY"
            # 차선 소실 즉시 비상 주행 로직 가동
            self.process_autonomous_driving(None) 

    #=============================================
    # 카메라 토픽을 수신하는 콜백 함수
    #=============================================
    def cam_callback(self, data):
        # 수신한 메시지를 OpenCV 이미지로 변환하여 저장
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    
    #=============================================
    # 라이다 토픽을 수신하는 콜백 함수
    #=============================================
    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges   
      
    #=============================================
    # 모터제어 토픽을 발행하는 Publisher 함수
    #=============================================
    def drive(self, angle, speed):
        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)
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

        look_ahead_index = int(image_height * look_ahead_ratio)
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
            # 1. EMERGENCY 상태 탈출 먼저 처리 (안전장치)
            if self.current_state == "EMERGENCY" and self.lane_valid_status and fit_x is not None:
                self.current_state = "STRAIGHT"

            # 2. 차선 소실 상태가 아니라면 '곡률 반경(R)' 기반으로 상태(FSM) 판단
            if self.current_state != "EMERGENCY" and fit_x is not None:
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

            if self.current_state == "STRAIGHT":
                # [직선] 멀리 보고(0.45), 매우 부드럽게 조향(0.015)하여 털림/지그재그 방지
                target_ratio = 0.45 
                target_Kp = 0.015  
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
                # 변화량 제한(Slew Rate) 없이, 계산된 각도를 즉시 핸들에 반영합니다.
                final_angle = self.calculate_angle(
                    fit_x, self.img_w, self.img_h, target_ratio, target_Kp, self.obstacle_offset)
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

