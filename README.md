# Custom Quadruped Robot Dog

This repository contains the design, simulation files, and code for a custom-built quadruped robot dog. It includes the Fusion 360 to URDF exports, meshes, and inverse kinematics resources (to be added).

## Repository Contents

*   `Robot_description/`: A standard ROS 2 package containing the URDF and simulation files exported from Fusion 360.
    *   `meshes/`: Contains all the 3D printable `.stl` files for the robot's parts.
    *   `urdf/`: Contains the `.urdf` model for the robot.
    *   `launch/`: Contains ROS 2 launch files.
        *   `display.launch.py`: Launches RViz to visualize the robot model.
        *   `gazebo.launch.py`: Launches Gazebo to simulate the robot in a physics environment.
    *   `config/`: Configuration files for RViz or controllers.

## How to Use the URDF Model

### Prerequisites
You will need a working installation of ROS 2 (e.g., Humble or Foxy) and standard packages like `robot_state_publisher`, `joint_state_publisher_gui`, `rviz2`, and `gazebo_ros`.

### Build the Package
1. Clone this repository into your ROS 2 workspace's `src` folder:
   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/YOUR_USERNAME/Robot-Dog.git
   ```
2. Build the workspace:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select Robot_description
   ```
3. Source the setup file:
   ```bash
   source install/setup.bash
   ```

### Visualization in RViz
To simply view the robot and move its joints manually:
```bash
ros2 launch Robot_description display.launch.py
```
This will open RViz with the robot model and a GUI to slide the joints.

### Simulation in Gazebo
To spawn the robot in a physics simulation:
```bash
ros2 launch Robot_description gazebo.launch.py
```

## Future Updates
*   **Inverse Kinematics**: Code for calculating IK for the legs.
*   **Control Scripts**: Walking gaits and stability algorithms.
*   **Media**: Pictures and videos of the physical build process.
