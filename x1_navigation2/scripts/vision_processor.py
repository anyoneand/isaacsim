#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs 
from image_geometry import PinholeCameraModel
# 【关键】引入 QoS 配置，必须有这一行才能接收 Isaac Sim 数据
from rclpy.qos import qos_profile_sensor_data

class VisionProcessor(Node):
    def __init__(self):
        super().__init__('vision_processor')

        # === 1. 初始化变量 (防止报错) ===
        self.bridge = CvBridge()
        self.camera_model = PinholeCameraModel()
        self.latest_depth_img = None
        self.camera_info_received = False 
        
        # === 2. 订阅话题 (加上 QoS) ===
        # RGB 图像
        self.create_subscription(
            Image, 
            '/camera/rgb', 
            self.rgb_callback, 
            qos_profile_sensor_data
        )
        
        # 深度图像
        self.create_subscription(
            Image, 
            '/camera/depth', 
            self.depth_callback, 
            qos_profile_sensor_data
        )
        
        # 相机内参
        self.create_subscription(
            CameraInfo, 
            '/camera/camera_info', 
            self.info_callback, 
            10
        )

        # === 3. TF 监听器 ===
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # === 4. 发布结果 ===
        self.target_pub = self.create_publisher(PointStamped, '/target_object_position', 10)

        self.get_logger().info("视觉 3D 定位节点已启动 (增强版)...")

    def info_callback(self, msg):
        if not self.camera_info_received:
            self.camera_model.fromCameraInfo(msg)
            self.camera_info_received = True
            self.get_logger().info(f"相机内参已接收, Frame ID: {msg.header.frame_id}")

    def depth_callback(self, msg):
        try:
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"深度图转换失败: {e}")

    def rgb_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # --- 【强制显示画面】用于调试 ---
            # 无论是否检测到，先弹窗，证明节点活着
            cv2.imshow("Vision Debug", cv_image)
            cv2.waitKey(1)
            # ---------------------------

            # 检查必要数据
            if not self.camera_info_received or self.latest_depth_img is None:
                if not self.camera_info_received and self.latest_depth_img is None:
                   self.get_logger().info("等待相机内参和深度图...", throttle_duration_sec=2)
                return

            # --- 【增强版】检测 AprilTag (支持镜像翻转) ---
            center, corners, is_flipped = self.detect_apriltag_robust(cv_image)
            
            if center is not None:
                u, v = center
                # 画绿圈标记中心
                cv2.circle(cv_image, (u, v), 8, (0, 255, 0), -1)
                
                # --- 获取深度 ---
                # 边界检查
                h, w = self.latest_depth_img.shape
                if 0 <= v < h and 0 <= u < w:
                    # 【关键修复】如果 RGB 是镜像检测的，深度图也要镜像采样
                    if is_flipped:
                        depth_source = cv2.flip(self.latest_depth_img, 1)
                    else:
                        depth_source = self.latest_depth_img

                    depth_val = depth_source[int(v), int(u)]
                    
                    if np.isfinite(depth_val) and depth_val > 0.0:
                        # 1. 2D -> 3D (相机坐标系)
                        ray = self.camera_model.projectPixelTo3dRay((u, v))
                        point_camera = np.array(ray) * depth_val
                        
                        # 2. 构建消息
                        p_cam_msg = PointStamped()
                        # 【关键】使用 Time(0) 获取最新 TF，防止仿真时间不同步报错
                        # 使用 0 时间戳表示 "最新的有效变换"
                        p_cam_msg.header.stamp = rclpy.time.Time(seconds=0).to_msg()
                        p_cam_msg.header.frame_id = self.camera_model.tfFrame()
                        p_cam_msg.point.x = point_camera[0]
                        p_cam_msg.point.y = point_camera[1]
                        p_cam_msg.point.z = point_camera[2]

                        # 3. 3D -> Base Link (或者 World)
                        try:
                            # 目标坐标系：建议直接转到 "base_link" (机器人基座) 
                            # 这样对于抓取规划更稳定，或者转到 "world" 也可以
                            target_frame = "base_link" 
                            
                            # transform 会自动查找最新的变换，因为我们设置了 stamp=0
                            p_world_msg = self.tf_buffer.transform(p_cam_msg, target_frame, timeout=rclpy.duration.Duration(seconds=1.0))
                            
                            self.target_pub.publish(p_world_msg)
                            self.get_logger().info(
                                f"定位成功! (镜像模式: {is_flipped})\n"
                                f"  Cam : X={point_camera[0]:.2f}, Y={point_camera[1]:.2f}, Z={point_camera[2]:.2f}\n"
                                f"  Base: X={p_world_msg.point.x:.2f}, Y={p_world_msg.point.y:.2f}, Z={p_world_msg.point.z:.2f}"
                            )
                        except Exception as e:
                            self.get_logger().warn(f"TF 变换出错: {e}")
            
        except Exception as e:
            self.get_logger().error(f"处理出错: {e}")

    def detect_apriltag_robust(self, img):
        """
        增强版检测：支持多种字典，支持左右镜像翻转
        返回: (center_coords, corners, is_flipped)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 定义要尝试的字典 (Isaac Sim 里的方块有时候是 36h11，有时候是 6x6)
        dictionaries_to_try = [
            cv2.aruco.DICT_APRILTAG_36h11,
            cv2.aruco.DICT_6X6_250
        ]

        # 2. 定义要尝试的翻转模式 (原图 + 水平翻转)
        # 你的截图显示纹理是镜像的，所以必须尝试 flip
        images_to_try = [
            (gray, "Normal"),
            (cv2.flip(gray, 1), "Flipped_Horizontal") 
        ]

        for dict_type in dictionaries_to_try:
            try:
                # 兼容不同版本的 OpenCV
                try:
                    dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
                except AttributeError:
                    dictionary = cv2.aruco.Dictionary_get(dict_type)

                try:
                    params = cv2.aruco.DetectorParameters()
                except AttributeError:
                    params = cv2.aruco.DetectorParameters_create()

                # 遍历原图和镜像图
                for img_curr, label in images_to_try:
                    corners, ids, _ = cv2.aruco.detectMarkers(img_curr, dictionary, parameters=params)
                    
                    if ids is not None and len(ids) > 0:
                        # 找到了！
                        c = corners[0][0]
                        center_u = int(np.mean(c[:, 0]))
                        center_v = int(np.mean(c[:, 1]))

                        is_flipped = (label == "Flipped_Horizontal")

                        if is_flipped:
                            self.get_logger().info(f"在 [镜像] 图像中检测到 Tag (ID={ids[0][0]}) - 使用修正坐标")
                        else:
                            self.get_logger().info(f"在 [正常] 图像中检测到 Tag (ID={ids[0][0]})")
                            # 只在非镜像时画框，不然框的位置会对不上
                            cv2.aruco.drawDetectedMarkers(img, corners, ids)

                        return (center_u, center_v), corners, is_flipped

            except Exception:
                continue

        return None, None, False

def main(args=None):
    rclpy.init(args=args)
    node = VisionProcessor()
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