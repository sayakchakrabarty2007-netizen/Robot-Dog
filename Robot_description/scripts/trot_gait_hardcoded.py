#!/usr/bin/env python3
"""
trot_gait.py  (v6 — IMU closed-loop yaw correction)

Diagonal-pair trot gait with real-time yaw drift correction using an IMU
sensor. The IMU provides the robot's orientation as a quaternion; we extract
the yaw angle, compare it to the initial heading captured at gait start,
and apply proportional-derivative (PD) differential steering to keep the
robot walking perfectly straight forever.

SETUP
-----
1. Robot.xacro / Robot.gazebo: imu_link + IMU sensor plugin publishing
   to Gazebo topic /imu/data

2. ros_gz_bridge must bridge the topic:
     ros2 run ros_gz_bridge parameter_bridge /imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU

3. This node subscribes to /imu/data (sensor_msgs/msg/Imu) on the ROS 2 side.
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import Imu
from builtin_interfaces.msg import Duration
import math
import time


# ======================= TUNABLE PARAMETERS ========================
CYCLE_PERIOD = 0.6            # full trot cycle (both diagonals), seconds
FEMUR_SWING_AMPLITUDE = 0.20  # radians, fore/aft swing range
KNEE_LIFT_AMPLITUDE = 0.40    # radians, knee lift during swing
STANCE_HEIGHT_COMP = 0.02     # smooth knee straightening on stance legs
CONTROL_RATE_HZ = 50.0        # Hz — trajectory update rate
SETTLE_TIME = 1.0             # seconds to hold standing pose before trotting
BLEND_DURATION = 2.0          # seconds to ramp gait amplitudes from 0 to full
BASE_HIP_SPRAWL = 0.05        # outward splay for wide stance
HIP_SWING_AMPLITUDE = 0.0     # outward arc during swing (0 = disabled)

# --- IMU Yaw PD Controller Gains ---
YAW_KP = 0.3                 # proportional gain: radians of yaw error → stride scale
YAW_KD = 0.05                # derivative gain: damps oscillation around heading
YAW_MAX_CORRECTION = 0.15    # clamp max differential steering to prevent over-correction

HIP_LATERAL_OFFSET = 0.075    # constant bias to push all legs left, sliding the body RIGHT to fix crab-walk
# ===================================================================


# Leg groups
LEFT_LEGS  = {'FL', 'BL'}
RIGHT_LEGS = {'FR', 'BR'}

# ---- Joint name mapping ----
JOINT_NAMES = {
    'FL': {'hip': 'Revolute 50', 'femur': 'Revolute 13', 'knee': 'Revolute 21'},
    'FR': {'hip': 'Revolute 51', 'femur': 'Revolute 15', 'knee': 'Revolute 17'},
    'BL': {'hip': 'Revolute 49', 'femur': 'Revolute 14', 'knee': 'Revolute 22'},
    'BR': {'hip': 'Revolute 48', 'femur': 'Revolute 16', 'knee': 'Revolute 18'},
}

# Sign conventions
FEMUR_SIGN   = {'FL': +1, 'BL': +1, 'FR': -1, 'BR': -1}
KNEE_SIGN    = {'FL': +1, 'BL': +1, 'FR': -1, 'BR': -1}
OUTWARD_SIGN = {'FL': +1, 'BL': +1, 'FR': -1, 'BR': -1}

# Diagonal pair phase offsets
TROT_OFFSETS = {'FL': 0.0, 'BR': 0.0, 'FR': 0.5, 'BL': 0.5}


# ------------ Utility functions -------------------

def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

def smoother_step(t):
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

def soft_bell(frac):
    return math.sin(math.pi * frac) ** 2

def quaternion_to_yaw(x, y, z, w):
    """
    Extract yaw from a Z-up ENU world frame quaternion.
    Gazebo's IMU plugin outputs orientation relative to the ENU world frame,
    so rotation around the vertical Z axis is always the heading.
    """
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)

def normalize_angle(angle):
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# -------------------------------------------------------------------

def compute_trot_leg(phase_01):
    """
    phase [0.0, 0.5) = SWING (leg in the air, moving forward)
    phase [0.5, 1.0) = STANCE (leg on the ground, pushing backward)
    """
    if phase_01 < 0.5:
        frac = phase_01 / 0.5
        femur = -FEMUR_SWING_AMPLITUDE * math.cos(frac * math.pi)
        knee = KNEE_LIFT_AMPLITUDE * soft_bell(frac)
        is_swing = True
    else:
        frac = (phase_01 - 0.5) / 0.5
        femur = FEMUR_SWING_AMPLITUDE - 2.0 * FEMUR_SWING_AMPLITUDE * smoother_step(frac)
        knee = 0.0
        is_swing = False
    return femur, knee, is_swing


class TrotGaitNode(Node):
    def __init__(self):
        super().__init__('trot_gait_node')

        # --- Publishers ---
        self.publisher_ = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # --- IMU Subscriber ---
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.start_time = time.time()
        self.gait_started = False
        self.timer = self.create_timer(1.0 / CONTROL_RATE_HZ, self.tick)

        # Only the 12 actuated joints
        self.joint_order = [
            'Revolute 48', 'Revolute 49', 'Revolute 50', 'Revolute 51',
            'Revolute 13', 'Revolute 14', 'Revolute 15', 'Revolute 16',
            'Revolute 17', 'Revolute 18', 'Revolute 21', 'Revolute 22',
        ]

        self.blend_factor = 0.0

        # --- IMU State ---
        self.current_yaw = 0.0          # latest yaw reading from IMU
        self.target_yaw = None          # captured at gait start (initial heading)
        self.prev_yaw_error = 0.0       # for derivative term
        self.imu_received = False       # flag: have we ever received IMU data?
        self.yaw_correction = 0.0       # current PD output (differential steering)

        self.get_logger().info(
            f"Trot gait v6 (IMU closed-loop) started: cycle={CYCLE_PERIOD}s, "
            f"settle={SETTLE_TIME}s, femur_amp={FEMUR_SWING_AMPLITUDE}rad, "
            f"knee_lift={KNEE_LIFT_AMPLITUDE}rad, "
            f"YAW PD: Kp={YAW_KP}, Kd={YAW_KD}, max={YAW_MAX_CORRECTION}"
        )
        self.get_logger().info("Waiting for IMU data on /imu/data ...")

    def imu_callback(self, msg: Imu):
        """Called every time a new IMU message arrives."""
        q = msg.orientation
        self.current_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        if not self.imu_received:
            self.imu_received = True
            self.get_logger().info(
                f"IMU data received! Initial yaw = {math.degrees(self.current_yaw):.1f}°"
            )

    def tick(self):
        t = time.time() - self.start_time

        # ---- Phase 0: Settling (hold standing pose) ----
        if t < SETTLE_TIME:
            self.publish_standing_pose()
            return

        # ---- Phase 1: Blend into gait ----
        if not self.gait_started:
            self.gait_started = True
            self.gait_start_time = time.time()
            # Capture current heading as the target
            self.target_yaw = self.current_yaw
            self.get_logger().info(
                f"Settling complete — starting trot. "
                f"Target heading locked: {math.degrees(self.target_yaw):.1f}°"
            )

        gait_t = time.time() - self.gait_start_time
        self.blend_factor = smoothstep(min(1.0, gait_t / BLEND_DURATION))

        # ---- IMU PD Yaw Controller ----
        if self.imu_received and self.target_yaw is not None:
            yaw_error = normalize_angle(self.current_yaw - self.target_yaw)

            # PD control
            d_error = yaw_error - self.prev_yaw_error
            correction = YAW_KP * yaw_error + YAW_KD * d_error
            self.prev_yaw_error = yaw_error

            # Clamp to prevent wild over-correction
            correction = max(-YAW_MAX_CORRECTION, min(YAW_MAX_CORRECTION, correction))
            self.yaw_correction = correction
        else:
            self.yaw_correction = 0.0

        # ---- Phase 2: Compute trot gait ----
        global_phase = (gait_t % CYCLE_PERIOD) / CYCLE_PERIOD

        commands = {}
        for leg_name, joints in JOINT_NAMES.items():
            local_phase = (global_phase + TROT_OFFSETS[leg_name]) % 1.0

            femur_c, knee_c, is_swing = compute_trot_leg(local_phase)

            hip_outward = 0.0

            if is_swing:
                swing_frac = local_phase / 0.5
                hip_outward = HIP_SWING_AMPLITUDE * math.sin(swing_frac * math.pi)
            else:
                swing_phase = (local_phase + 0.5) % 1.0
                swing_frac = swing_phase / 0.5
                knee_c += -STANCE_HEIGHT_COMP * math.sin(swing_frac * math.pi)

            # Blend from standing
            femur_c *= self.blend_factor
            knee_c  *= self.blend_factor

            # Hip: permanent sprawl + lateral anti-crab offset
            # A positive HIP_LATERAL_OFFSET moves all legs LEFT, pushing the body RIGHT
            sprawl_c = BASE_HIP_SPRAWL + hip_outward
            hip_c = (OUTWARD_SIGN[leg_name] * sprawl_c) + HIP_LATERAL_OFFSET
            hip_c *= self.blend_factor

            # Closed-loop Differential Steering from IMU
            # If robot has turned LEFT (positive yaw error), we need to steer RIGHT:
            #   → Left legs take BIGGER strides, Right legs take SMALLER strides
            # self.yaw_correction is positive when robot has turned left
            if leg_name in LEFT_LEGS:
                stride_scale = 1.0 + self.yaw_correction
            else:
                stride_scale = 1.0 - self.yaw_correction

            commands[joints['femur']] = FEMUR_SIGN[leg_name] * (femur_c * stride_scale)
            commands[joints['knee']]  = KNEE_SIGN[leg_name]  * knee_c
            commands[joints['hip']]   = hip_c

        self.publish_commands(commands)

    def publish_standing_pose(self):
        commands = {name: 0.0 for name in self.joint_order}
        self.publish_commands(commands)

    def publish_commands(self, commands):
        msg = JointTrajectory()
        msg.joint_names = list(self.joint_order)

        point = JointTrajectoryPoint()
        point.positions = [commands.get(name, 0.0) for name in self.joint_order]
        dt_ns = int(1e9 / CONTROL_RATE_HZ * 2)
        point.time_from_start = Duration(sec=0, nanosec=dt_ns)

        msg.points.append(point)
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrotGaitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()