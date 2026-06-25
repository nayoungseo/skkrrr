#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Bool
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data



class LaneDetectionNode(Node):

    def __init__(self):

        super().__init__('lane_detection')

        self.get_logger().info(
            "===== Lane Detection Node Started ====="
        )

        self.bridge = CvBridge()

        # 주행 시 시각화 창을 켤지 말지 결정하는 변수 (실전 주행 시 False로 변경)
        self.DEBUG = True
        self.last_fit_x = None
        self.lost_frame_count = 0
        self.cone_mode_active = True

        ##################################################
        # SUBSCRIBE
        ##################################################

        self.sub_cam = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.cam_callback,
            qos_profile_sensor_data
        )

        self.sub_lane_switch = self.create_subscription(
            Bool,
            '/vision/cone_valid',
            self.lane_switch_callback,
            10
        )

        ##################################################
        # PUBLISH
        ##################################################

        self.fit_x_pub = self.create_publisher(
            Float32MultiArray,
            '/vision/lane_fit_x',
            10
        )

        self.valid_pub = self.create_publisher(
            Bool,
            '/vision/lane_valid',
            10
        )

    def lane_switch_callback(self, msg):
        if not msg.data and self.cone_mode_active:
            self.cone_mode_active = False
            self.get_logger().info(
                "Cone handoff complete. Camera lane node takes control."
            )

    ##################################################
    # CAMERA CALLBACK
    ##################################################

    def cam_callback(self, msg):

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                "bgr8"
            )

            fit_x, valid = self.process_lane(frame)

            if self.cone_mode_active:
                valid_msg = Bool()
                valid_msg.data = valid
                self.valid_pub.publish(valid_msg)
                return

            if valid and fit_x is not None:
                self.last_fit_x = fit_x
                self.lost_frame_count = 0
            else:
                self.lost_frame_count += 1
                if self.last_fit_x is not None:
                    fit_x = self.last_fit_x
                    valid = True
                else:
                    fit_x = self.center_fallback_fit(frame)
                    valid = True

            ##################################################
            # publish fit_x
            ##################################################
            if valid and fit_x is not None:
                fit_x_msg = Float32MultiArray()
                # numpy 배열을 파이썬 기본 list로 변환하여 담아줍니다.
                fit_x_msg.data = fit_x.tolist()
                self.fit_x_pub.publish(fit_x_msg)

            ##################################################
            # publish validity
            ##################################################

            valid_msg = Bool()
            valid_msg.data = valid

            self.valid_pub.publish(valid_msg)

        except Exception as e:

            self.get_logger().warn(
                f"Lane callback failed : {e}"
            )

            valid_msg = Bool()
            valid_msg.data = False

            # [수정 고침] 예외 발생 시에도 유효성 False 토픽을 명확히 전달
            self.valid_pub.publish(valid_msg)

    def center_fallback_fit(self, frame):
        h, w = frame.shape[:2]
        return np.full(h, w // 2, dtype=np.float32)

    def center_fallback_fit(self, frame):
        h, w = frame.shape[:2]
        return np.full(h, w // 2, dtype=np.float32)

    ##################################################
    # MAIN PIPELINE
    ##################################################

    def process_lane(self, frame):

        ##################################################
        # 1 Calibration
        ##################################################

        calibrated = self.calibration(frame)

        ##################################################
        # 2 Bird Eye View
        ##################################################

        bird = self.bird_eye(calibrated)

        ##################################################
        # 3 HSV Yellow Extraction
        ##################################################

        mask = self.yellow_extract(bird)

        ##################################################
        # 4 Sliding Window
        ##################################################

        fix_x, debug_img = self.sliding_window(mask)

        if fix_x is None:
            return None, False

        return fix_x, True


    ##################################################
    # CALIBRATION
    ##################################################

    def calibration(self, image):

        """
        현재 calibration 정보 없음.

        placeholder.

        나중에 camera matrix 넣을 위치.
        """

        return image

    ##################################################
    # BIRD EYE VIEW
    ##################################################

    def bird_eye(self, image):

        h, w = image.shape[:2]

        src = np.float32([

            [260,265],
            [380,265],

            [20,470],
            [620,470]

        ])

        dst = np.float32([

            [150,0],
            [490,0],

            [150,480],
            [490,480]

        ])

        M = cv2.getPerspectiveTransform(
            src,
            dst
        )

        warped = cv2.warpPerspective(
            image,
            M,
            (w,h)
        )

        ##버드아이뷰값 확인
        # debug = image.copy()

        # # ROI 사다리꼴 그리기
        # cv2.polylines(
        #     debug,
        #     [src.astype(np.int32)],
        #     True,
        #     (0,255,0),
        #     3
        # )

        # # 꼭짓점 표시
        # for p in src.astype(int):

        #     cv2.circle(
        #         debug,
        #         tuple(p),
        #         8,
        #         (0,0,255),
        #         -1
        #     )

        # combined = np.hstack([

        #     cv2.resize(debug,(640,480)),
        #     cv2.resize(warped,(640,480))

        # ])

        # cv2.imshow(
        #     "LEFT: ROI | RIGHT: BirdEye",
        #     combined
        # )

        # cv2.waitKey(1)

        # return warped

        return warped

    ##################################################
    # HSV YELLOW EXTRACTION
    ##################################################

    def yellow_extract(self, image):

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        lower_yellow = np.array([
            15,
            80,
            80
        ])

        upper_yellow = np.array([
            40,
            255,
            255
        ])

        mask = cv2.inRange(
            hsv,
            lower_yellow,
            upper_yellow
        )

        kernel = np.ones((5,5),np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # ##################################################
        # # AREA FILTER (YELLOW VEHICLE REMOVE)
        # ##################################################

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        filtered_mask = np.zeros_like(mask)

        ##################################################
        ##### ===== TUNING REQUIRED ===== #####
        ##################################################

        MAX_AREA = 4000

        ##################################################

        for cnt in contours:

            area = cv2.contourArea(cnt)

            # 디버깅용
            # print("AREA =", area)

            if area > MAX_AREA:

                continue

            cv2.drawContours(
                filtered_mask,
                [cnt],
                -1,
                255,
                -1
            )

        # #컨투어 확인
        # debug = cv2.cvtColor(
        #     filtered_mask,
        #     cv2.COLOR_GRAY2BGR
        # )

        # cv2.drawContours(
        #     debug,
        #     contours,
        #     -1,
        #     (0,0,255),
        #     2
        # )

        # cv2.imshow("CONTOURS", debug)

        #  # yellow 확인
        # mask_color = cv2.cvtColor(
        #     filtered_mask,
        #     cv2.COLOR_GRAY2BGR
        # )

        # combined = np.hstack([

        #     cv2.resize(image,(640,480)),
        #     cv2.resize(mask_color,(640,480))

        # ])

        # cv2.imshow(
        #     "LEFT: BirdEye | RIGHT: Yellow Mask",
        #     combined
        # )

        # cv2.waitKey(1)

        return filtered_mask


    ##################################################
    # SLIDING WINDOW
    ##################################################

    def sliding_window(self, binary_img):

        h, w = binary_img.shape

        debug_img = cv2.cvtColor(
            binary_img,
            cv2.COLOR_GRAY2BGR
        )

        ##################################################
        # HISTOGRAM START
        ##################################################

        histogram = np.sum(
            binary_img[h//2:,:],
            axis=0
        )

        current_x = np.argmax(
            histogram
        )

        ##################################################
        ##### ===== TUNING REQUIRED ===== #####
        ##################################################

        window_height = 60
        overlap = 30
        margin = 200
        minpix = 15

        ##################################################

        lane_x = []
        lane_y = []

        nonzero = binary_img.nonzero()

        nonzero_y = np.array(
            nonzero[0]
        )

        nonzero_x = np.array(
            nonzero[1]
        )

        y_top = h-window_height

        while y_top > 0:

            y_bottom = y_top+window_height

            x_left = current_x-margin

            x_right = current_x+margin

            ##################################################
            # DRAW WINDOW
            ##################################################

            cv2.rectangle(

                debug_img,

                (x_left,y_top),
                (x_right,y_bottom),

                (0,255,0),
                2
            )

            ##################################################
            # FIND PIXELS
            ##################################################

            good_inds=(

                (nonzero_y>=y_top)&
                (nonzero_y<y_bottom)&
                (nonzero_x>=x_left)&
                (nonzero_x<x_right)

            ).nonzero()[0]

            if len(good_inds)>minpix:

                current_x = int(np.mean(nonzero_x[good_inds]))

                current_y = int((y_top + y_bottom) / 2)

                lane_x.append(current_x)
                lane_y.append(current_y)

                # 차선 확인
                cv2.circle(
                    debug_img, 
                    (current_x, current_y), 
                    5,            # 반지름
                    (255, 0, 0),  # BGR 기준 파란색 (Blue)
                    -1            # 내부 채우기
                )

            ##################################################
            # OVERLAP STEP
            ##################################################

            y_top -= (window_height-overlap)

        ##################################################
        # RESULT
        ##################################################

        if len(lane_x) < 2:
            return None, debug_img

        centers_x = np.array(lane_x)
        centers_y = np.array(lane_y)

        # 점이 3개 이상이면 정상적인 2차 곡선 피팅
        if len(lane_x) >= 3:
            poly_coefficients = np.polyfit(centers_y, centers_x, 2)
        # 점이 딱 2개뿐이면 경고(Warning)를 방지하기 위해 1차 직선 피팅 후 형식 매칭
        else:
            poly_coefficients_1d = np.polyfit(centers_y, centers_x, 1)
            poly_coefficients = np.array([0, poly_coefficients_1d[0], poly_coefficients_1d[1]])
        
        plot_y = np.linspace(0, h - 1, h)

        fit_x = (poly_coefficients[0] * (plot_y ** 2) + 
                poly_coefficients[1] * plot_y + 
                poly_coefficients[2])

        pts = np.vstack((fit_x, plot_y)).astype(np.int32).T

        cv2.polylines(debug_img, [pts], isClosed=False, color=(0, 255, 255), thickness=3)
        

        if self.DEBUG:
            cv2.imshow("SLIDING WINDOW DEBUG", debug_img)
            cv2.waitKey(1)

        return fit_x, debug_img



##################################################
# MAIN
##################################################

def main(args=None):

    rclpy.init(args=args)

    node = LaneDetectionNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        cv2.destroyAllWindows()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__=="__main__":

    main()
