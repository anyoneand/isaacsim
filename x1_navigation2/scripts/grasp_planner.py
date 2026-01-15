#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState as ROSJointState
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import torch
import copy
import time

# QoS 配置 (关键修复)
from rclpy.qos import qos_profile_sensor_data

# CuRobo imports
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState as CuroboJointState
from curobo.util_file import load_yaml

class DualArmCooperativePlanner(Node):
    def __init__(self):
        super().__init__('dual_arm_cooperative_planner')
        self.get_logger().info("Initializing Dual Arm Cooperative Planner (Fixed Version)...")

        # --- Configuration ---
        self.config_file = "/home/js/Downloads/src/isaacsim_python-second_competition/src/x1_navigation2/config/x1_curobo.yaml"
        self.robot_base_frame = "base_link"
        self.dt = 1.0 / 30.0

        # --- State Machine ---
        self.state = "IDLE" 
        
        # --- ROS Communication ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub_target = self.create_subscription(
            PointStamped, '/target_object_position', self.target_callback, 10)

        self.sub_joints = self.create_subscription(
            ROSJointState, 
            '/joint_states', 
            self.joint_state_callback, 
            qos_profile_sensor_data
        )

        self.pub_cmd = self.create_publisher(ROSJointState, '/joint_command', 10)

        self.timer = self.create_timer(self.dt, self.control_loop)

        # --- CuRobo Init ---
        self.setup_curobo()

        # Cache
        self.current_joints_map = {} 
        self.plan_queue = []
        self.traj_index = 0
        self.current_trajectory = None

        # Start Homing Sequence
        self.trigger_homing = True 

    def setup_curobo(self):
        self.get_logger().info("Setting up CuRobo...")
        self.tensor_args = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)
        base_cfg = load_yaml(self.config_file)["robot_cfg"]
        
        # Standard Home Pose (Forward Ready - Revised)
        # J2 positive = Forward/Down? J3 Negative = Up/In? 
        # Tuning based on user report of "backward" motion with negative J2.
        self.home_pose = [0.3, 0.5, -1.0, -0.5, 0.0, 0.0, 0.0] 
        self.home_pose_R = [-0.3, 0.5, -1.0, -0.5, 0.0, 0.0, 0.0] 

        # 1. Left Arm
        cfg_left = copy.deepcopy(base_cfg)
        cfg_left["kinematics"]["ee_link"] = "link7_L"
        cfg_left["kinematics"]["cspace"]["joint_names"] = ["joint1_L", "joint2_L", "joint3_L", "joint4_L", "joint5_L", "joint6_L", "joint7_L"]
        cfg_left["kinematics"]["base_link"] = "link_body"
        # Normal (Safe) Generator
        self.mg_left = self._create_motion_gen(cfg_left, check_collision=True)
        # Loose (Recovery) Generator
        self.mg_left_loose = self._create_motion_gen(cfg_left, check_collision=False)

        # 2. Right Arm
        cfg_right = copy.deepcopy(base_cfg)     
        cfg_right["kinematics"]["ee_link"] = "link7_R"
        cfg_right["kinematics"]["cspace"]["joint_names"] = ["joint1_R", "joint2_R", "joint3_R", "joint4_R", "joint5_R", "joint6_R", "joint7_R"]
        cfg_right["kinematics"]["base_link"] = "link_body"
        # Normal (Safe) Generator
        self.mg_right = self._create_motion_gen(cfg_right, check_collision=True)
        # Loose (Recovery) Generator
        self.mg_right_loose = self._create_motion_gen(cfg_right, check_collision=False)
        
        self.plan_config = MotionGenPlanConfig(enable_graph=True, max_attempts=10)
        self.get_logger().info("System Ready!")

    def _create_motion_gen(self, robot_cfg, check_collision=True):
        mg_config = MotionGenConfig.load_from_robot_config(
            robot_cfg, None, self.tensor_args,
            trajopt_tsteps=32, use_cuda_graph=True,
            interpolation_dt=self.dt, self_collision_check=check_collision 
        )
        mg = MotionGen(mg_config)
        mg.warmup(enable_graph=True)
        return mg

    def joint_state_callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joints_map[name] = pos

    def get_current_joints(self, arm="left"):
        mg = self.mg_left if arm == "left" else self.mg_right
        vals = []
        
        for i, n in enumerate(mg.kinematics.joint_names):         
            if "joint7" in n:
                vals.append(0.0)
                continue         
            if n not in self.current_joints_map: 
                return None
            vals.append(self.current_joints_map[n])
        return np.array(vals, dtype=np.float32)

    def target_callback(self, msg):
        if not self.current_joints_map: return

        if self.trigger_homing:
            if self.state == "IDLE":
                self.get_logger().info(">>> FORCING HOMING SEQUENCE <<<")
                if self.perform_homing():
                    self.trigger_homing = False
            return

        if self.state == "IDLE":
            try:
                target_base = self.tf_buffer.transform(msg, self.robot_base_frame, timeout=Duration(seconds=1.0))
                self.plan_dual_grasp(target_base)
            except Exception as e:
                self.get_logger().error(f"Target error: {e}")

    def perform_homing(self):
        self.state = "HOMING"
        start_L = self.get_current_joints("left")
        start_R = self.get_current_joints("right")
        
        if start_L is None or start_R is None:
            self.get_logger().error("Homing Fail: Missing joint states.")
            self.state = "IDLE"; return False
            
        self.get_logger().info(">>> PERFORMING BLIND HOMING (Forcing Reset) <<<")
        self.get_logger().info(f"Start L: {np.round(start_L, 2)}")
        
        # Generate Blind Trajectory (Linear Interpolation)
        # 150 steps @ 30Hz = 5 seconds
        steps = 150
        traj = []
        
        goal_L = np.array(self.home_pose)
        goal_R = np.array(self.home_pose_R)
        
        for i in range(steps):
            alpha = float(i) / (steps - 1)
            # Smoothstep for nicer motion (t * t * (3 - 2 * t))
            t = alpha
            smooth_t = t * t * (3 - 2 * t)
            
            pos_L = start_L + (goal_L - start_L) * smooth_t
            pos_R = start_R + (goal_R - start_R) * smooth_t
            
            # Combine 12 DOF (Left + Right)
            traj.append(np.concatenate([pos_L, pos_R]).tolist())
            
        self.plan_queue = [traj]
        self.start_next_trajectory()
        self.get_logger().info("Blind Homing Trajectory Generated.")
        return True

    def plan_dual_grasp(self, target_base):
        self.state = "PLANNING"
        tx, ty, tz = target_base.point.x, target_base.point.y, target_base.point.z
        
        # Funnel Strategy
        y_pre, y_grasp, z_pre = 0.20, 0.08, tz + 0.25
        quat_down = np.array([0.0, 1.0, 0.0, 0.0]) # w, x, y, z

        # Plan sequences for both arms
        try:
            # --- Left Arm ---
            current_L = self.get_current_joints("left")
            t_pre_L = self.plan_single(self.mg_left, current_L, [tx, ty+y_pre, z_pre], quat_down)
            
            # Extract end state from Pre-Grasp to use as Start for Grasp
            start_grasp_L = None
            if t_pre_L is not None:
                # JointState position is tensor [batch, steps, dof] or [steps, dof]
                # We need the last step as a flat list
                last_pos = t_pre_L.position[-1] # Get last step
                start_grasp_L = last_pos.cpu().numpy().flatten().tolist()
            
            t_grasp_L = self.plan_single(self.mg_left, start_grasp_L, [tx, ty+y_grasp, tz], quat_down)
            
            # --- Right Arm ---
            current_R = self.get_current_joints("right")
            t_pre_R = self.plan_single(self.mg_right, current_R, [tx, ty-y_pre, z_pre], quat_down)
            
            start_grasp_R = None
            if t_pre_R is not None:
                last_pos = t_pre_R.position[-1]
                start_grasp_R = last_pos.cpu().numpy().flatten().tolist()

            t_grasp_R = self.plan_single(self.mg_right, start_grasp_R, [tx, ty-y_grasp, tz], quat_down)

            if all([t_pre_L, t_grasp_L, t_pre_R, t_grasp_R]):
                self.plan_queue = [
                    self.merge_trajectories(t_pre_L, t_pre_R),
                    self.merge_trajectories(t_grasp_L, t_grasp_R)
                ]
                self.start_next_trajectory()
            else:
                self.get_logger().error("Planning failed for one or more segments")
                self.state = "IDLE"
        except Exception as e:
            self.get_logger().error(f"Grasp Plan Error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.state = "IDLE"

    def plan_single(self, mg, start_pos, target_xyz, target_quat):
        if start_pos is None: return None
        
        js_start = CuroboJointState.from_position(
            torch.as_tensor(start_pos, **self.tensor_args.as_torch_dict()).reshape(1, -1),
            joint_names=mg.kinematics.joint_names
        )
        goal_pose = Pose(
            position=torch.as_tensor(target_xyz, **self.tensor_args.as_torch_dict()).reshape(1, 3),
            quaternion=torch.as_tensor(target_quat, **self.tensor_args.as_torch_dict()).reshape(1, 4)
        )
        
        res = mg.plan_single(js_start, goal_pose, self.plan_config)
        return res.get_interpolated_plan() if res.success.item() else None

    def merge_trajectories(self, traj_L, traj_R):
        # Handle cases where one side might be just a list or a CuRobo object
        pos_L = traj_L.position.cpu().numpy() if hasattr(traj_L, 'position') else np.array(traj_L)
        pos_R = traj_R.position.cpu().numpy() if hasattr(traj_R, 'position') else np.array(traj_R)
        
        max_len = max(len(pos_L), len(pos_R))
        combined = []
        for i in range(max_len):
            pL = pos_L[min(i, len(pos_L)-1)].flatten().tolist()
            pR = pos_R[min(i, len(pos_R)-1)].flatten().tolist()
            combined.append(pL + pR)
        return combined

    def start_next_trajectory(self):
        if not self.plan_queue:
            self.state = "IDLE"; return
        self.current_trajectory = self.plan_queue.pop(0)
        self.traj_index = 0
        self.state = "EXECUTING"
        joint_names = self.mg_left.kinematics.joint_names + self.mg_right.kinematics.joint_names
        self.get_logger().info(f"Executing segment... ({len(self.current_trajectory)} points)")
        self.get_logger().info(f"PUBLISHING JOINTS: {joint_names}")

    def control_loop(self):
        if self.state != "EXECUTING" or not self.current_trajectory: return
        if self.traj_index >= len(self.current_trajectory):
            self.start_next_trajectory(); return
            
        msg = ROSJointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        full_trajectory_point = self.current_trajectory[self.traj_index] # 14个值
        
        left_vals = full_trajectory_point[0:7]
        right_vals = full_trajectory_point[7:14]
        
        left_cmd = left_vals[0:6]
        right_cmd = right_vals[0:6]
        
        ros_names = [
            "joint1_L", "joint2_L", "joint3_L", "joint4_L", "joint5_L", "joint6_L",
            "joint1_R", "joint2_R", "joint3_R", "joint4_R", "joint5_R", "joint6_R"
        ]
        msg.name = ros_names
        msg.position = left_cmd + right_cmd 
        self.pub_cmd.publish(msg)
        self.traj_index += 1

def main():
    rclpy.init()
    node = DualArmCooperativePlanner()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()
