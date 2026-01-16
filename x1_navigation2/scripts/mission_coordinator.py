#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
import subprocess
import time
import math
import signal

class MissionCoordinator(Node):
    def __init__(self):
        super().__init__('mission_coordinator')
        
        self.cb_group = ReentrantCallbackGroup()
        
        # --- State ---
        self.state = "IDLE" 
        self.vision_process = None
        self.grasp_process = None
        self.task_start_time = 0
        
        self.current_pose = None
        self.target_pose = None
        self._nav_goal_future = None
        
        # --- Config ---
        self.vision_cmd = ["ros2", "run", "x1_navigation2", "vision_processor.py"] 
        self.grasp_cmd = ["ros2", "run", "x1_navigation2", "grasp_planner.py"]
        
        # --- ROS Interface ---
        self.sub_goal = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10, callback_group=self.cb_group)
            
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10, callback_group=self.cb_group)
            
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose', callback_group=self.cb_group)
        
        self.sub_grasp_status = self.create_subscription(
            String, '/grasp_status', self.grasp_status_callback, 10, callback_group=self.cb_group)

        # Polling Timer
        self.nav_poll_timer = self.create_timer(0.5, self.check_nav_status, callback_group=self.cb_group)

        self.get_logger().info(">>> MISSION COORDINATOR READY (Distance Check Mode) <<<")
        self.get_logger().info("Please set a 2D Nav Goal in Rviz.")

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def goal_callback(self, msg):
        if self.state != "IDLE":
            self.get_logger().warn("Busy! Ignoring new goal.")
            return
        
        self.target_pose = msg.pose
        self.get_logger().info(f"Received Goal! Navigating...")
        self.send_nav_goal(msg)

    def send_nav_goal(self, pose_msg):
        self.state = "NAVIGATING"
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 Action Server missing!")
            self.state = "IDLE"
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_msg
        
        self._nav_goal_future = self._nav_client.send_goal_async(goal_msg)

    def check_nav_status(self):
        # 1. Check Goal Acceptance (Just for logging)
        if self._nav_goal_future and self._nav_goal_future.done():
            goal_handle = self._nav_goal_future.result()
            self._nav_goal_future = None 
            if not goal_handle.accepted:
                self.get_logger().error('Goal rejected.')
                self.state = "IDLE"
                return
            self.get_logger().info('Nav2 accepted goal. Moving...')

        # 2. DISTANCE CHECK (The real trigger)
        if self.state == "NAVIGATING" and self.current_pose and self.target_pose:
            dx = self.current_pose.position.x - self.target_pose.position.x
            dy = self.current_pose.position.y - self.target_pose.position.y
            dist = math.sqrt(dx*dx + dy*dy)
            
            # Tolerance: 0.15m (matches Nav2 general checker)
            if dist < 0.15:
                self.get_logger().info(f'>>> TARGET REACHED (Dist: {dist:.2f}m) <<<')
                # Optional: Cancel Nav2 goal to stop it trying to refine perfectly?
                # self._nav_client._cancel_goal_async(...)
                self.start_manipulation_task()

    def start_manipulation_task(self):
        if self.state == "WORKING": return
        self.state = "WORKING"
        self.task_start_time = time.time()
        
        try:
            self.get_logger().info(f"Launching Vision: {' '.join(self.vision_cmd)}")
            self.vision_process = subprocess.Popen(
                self.vision_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=0)
            
            time.sleep(3.0)
            
            self.get_logger().info(f"Launching Grasp: {' '.join(self.grasp_cmd)}")
            self.grasp_process = subprocess.Popen(
                self.grasp_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=0)
            
            self.timer = self.create_timer(1.0, self.check_task_timeout, callback_group=self.cb_group)
        except Exception as e:
            self.get_logger().error(f"LAUNCH FAILED: {e}")
            self.state = "IDLE"

    # ... (Rest of callbacks same as before)
    def grasp_status_callback(self, msg):
        if self.state != "WORKING": return
        if msg.data == "DONE" or msg.data == "FAILED":
            self.get_logger().info(f"Grasp Task Reported: {msg.data}")
            self.stop_manipulation_task()

    def check_task_timeout(self):
        if self.state != "WORKING": return
        elapsed = time.time() - self.task_start_time
        if elapsed > 60.0:
            self.get_logger().warn("Task Timed Out!")
            self.stop_manipulation_task()

    def stop_manipulation_task(self):
        self.get_logger().info("Stopping Task...")
        if self.vision_process:
            self.vision_process.send_signal(signal.SIGINT)
            self.vision_process = None
        if self.grasp_process:
            self.grasp_process.send_signal(signal.SIGINT)
            self.grasp_process = None
        if hasattr(self, 'timer'): self.timer.cancel()
        self.state = "IDLE"
        self.get_logger().info(">>> READY FOR NEXT GOAL <<<")

def main(args=None):
    rclpy.init(args=args)
    node = MissionCoordinator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_manipulation_task()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
