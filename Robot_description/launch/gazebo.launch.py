from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
import os
import xacro
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share_dir = get_package_share_directory('Robot_description')

    xacro_file = os.path.join(share_dir, 'urdf', 'Robot.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    robot_urdf = robot_description_config.toxml()

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'robot_description': robot_urdf}
        ]
    )

    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher'
    )

    # Replaced gazebo_ros with ros_gz_sim for ROS 2 Jazzy
    gazebo_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        # Just passing empty.sdf without '-r' ensures it starts paused!
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    # Replaced spawn_entity.py with 'create' for Gazebo Sim
    urdf_spawn_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'Robot',
            '-topic', 'robot_description',
            '-z', '0.6',  # Spawns the robot 0.6 meters in the air
            '-R', '1.5708'  # <--- THIS ROTATES THE WHOLE ROBOT 90 DEGREES ON SPAWN
        ],
        output='screen'
    )

    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_node,
        gazebo_sim,
        urdf_spawn_node,
    ])