import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Paths
    x1_navigation2_dir = get_package_share_directory('x1_navigation2')
    nav2_launch_file = os.path.join(x1_navigation2_dir, 'launch', 'navigation2.launch.py')
    
    # Args
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo/Isaac) clock if true'),

        # 1. Launch Navigation2 Stack (Nav2 + Rviz)
        # This will load the map, amcl, controller, planner, and rviz
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch_file),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),

        # 2. Launch Mission Coordinator Node
        # This node listens for Rviz goals and coordinates Vision/Grasping
        Node(
            package='x1_navigation2',
            executable='mission_coordinator.py',
            name='mission_coordinator',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}]
        ),
    ])
