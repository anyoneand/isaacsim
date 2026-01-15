#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class AprilTagDetector(Node):
    def __init__(self):
        super().__init__('apriltag_detector')
        
        # 1. 创建订阅者，订阅 /camera/rgb 话题
        # 注意：这里的话题名必须和你之前 'ros2 topic list' 查看到的一致
        self.subscription = self.create_subscription(
            Image,
            '/camera/rgb', 
            self.listener_callback,
            10)
        
        self.cv_bridge = CvBridge()
        self.get_logger().info('AprilTag 检测节点已启动，正在监听 /camera/rgb ...')

    def listener_callback(self, msg):
        try:
            # 2. 将 ROS 图像消息转换为 OpenCV 图像
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # 3. 调用你的检测函数
            corners, center = self.detect_aprilTag(cv_image, show_result_img=True)
            
            if center is not None:
                self.get_logger().info(f"检测到 AprilTag! 中心坐标: {center}")
            
        except Exception as e:
            self.get_logger().error(f'图像处理出错: {e}')

    # --- 这里是你提供的代码逻辑 (已补全) ---
    def detect_aprilTag(self, rgb_img=None, show_result_img=False):
        # 添加图像输入校验
        if rgb_img is None:
            self.get_logger().warn("未输入图像，无法检测AprilTag")
            return None, None

        # 转灰度
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        
        # 获取字典和参数
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        # 尝试兼容不同版本的 OpenCV
        try:
            # OpenCV 4.7+ 写法
            parameters = cv2.aruco.DetectorParameters()
        except AttributeError:
            # OpenCV 4.5/4.6 写法 (ROS Humble 默认)
            parameters = cv2.aruco.DetectorParameters_create()
        
        # 检测
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
        
        tag_corners = None
        center_coords = None

        if ids is not None and len(ids) > 0:
            # 获取第一个检测到的标签
            tag_corners = corners[0]
            
            # 计算中心点 (corners 是一个 1x4x2 的数组)
            # 这里的逻辑是取四个角的平均值作为中心
            c = tag_corners[0]
            center_u = int(np.mean(c[:, 0]))
            center_v = int(np.mean(c[:, 1]))
            center_coords = [center_u, center_v]

            # 可视化（如果需要）
            if show_result_img:
                cv2.aruco.drawDetectedMarkers(rgb_img, corners, ids)
                cv2.circle(rgb_img, (center_u, center_v), 5, (0, 0, 255), -1)
                cv2.imshow("AprilTag Detection", rgb_img)
                cv2.waitKey(1)
        else:
            if show_result_img:
                cv2.imshow("AprilTag Detection", rgb_img)
                cv2.waitKey(1)

        return tag_corners, center_coords

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()