import os
import launch
import launch_ros
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # ... (前面的代码保持不变) ...
    x1_navigation2_dir = get_package_share_directory('x1_navigation2')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    rviz_config_dir = os.path.join(nav2_bringup_dir, 'rviz', 'nav2_default_view.rviz')
    
    use_sim_time = launch.substitutions.LaunchConfiguration('use_sim_time', default='true')
    map_yaml_path = launch.substitutions.LaunchConfiguration('map', default=os.path.join(x1_navigation2_dir, 'maps', 'carter_warehouse_navigation.yaml'))
    nav2_param_path = launch.substitutions.LaunchConfiguration('params_file', default=os.path.join(x1_navigation2_dir, 'config', 'nav2_params.yaml'))

    return launch.LaunchDescription([
        # ... (声明参数的部分保持不变) ...
        launch.actions.DeclareLaunchArgument('use_sim_time', default_value=use_sim_time, description='Use simulation (Gazebo) clock if true'),
        launch.actions.DeclareLaunchArgument('map', default_value=map_yaml_path, description='Full path to map file to load'),
        launch.actions.DeclareLaunchArgument('params_file', default_value=nav2_param_path, description='Full path to param file to load'),

        # 包含 Nav2 的启动文件
        launch.actions.IncludeLaunchDescription(
            PythonLaunchDescriptionSource([nav2_bringup_dir, '/launch', '/bringup_launch.py']),
            launch_arguments={
                'map': map_yaml_path,
                'use_sim_time': use_sim_time,
                'params_file': nav2_param_path}.items(),
        ),

        # 启动 Rviz2
        launch_ros.actions.Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_dir],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'),

        # ==========================================
        # 【新增】 在这里添加你的 AprilTag 检测节点
        # ==========================================
        # 注意：
        # 1. package 填你放置 apriltag_node.py 的包名 (假设是 'x1_navigation2')
        # 2. executable 填你在 setup.py 中配置的入口点名称
        #    如果只是测试，也可以不加在这里，而是另外开一个终端直接运行 python 脚本
        launch_ros.actions.Node(
             package='x1_navigation2', # 请修改为你的实际包名
             executable='apriltag_node.py', # 请修改为你在 setup.py 配置的可执行名
             name='apriltag_detector',
             output='screen',
        ),
    ])