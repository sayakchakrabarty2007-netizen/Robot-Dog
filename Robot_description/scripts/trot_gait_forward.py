#!/usr/bin/env python3
"""
trot_gait.py  (v7 — Inverse Kinematics trot with IMU closed-loop yaw)

Instead of hardcoding femur/knee oscillator amplitudes, this version defines
foot trajectories in Cartesian space (x, y, z) relative to each hip, then
uses the leg_kinematics.leg_ik_to_command() function to convert them to
joint angles. This is much more intuitive and physically correct.

GAIT DESIGN
-----------
Each leg's foot traces a closed loop in the leg-local XZ plane:
  - SWING (airborne): foot follows an elliptical arc — forward and UP
  - STANCE (ground):  foot slides straight backward at constant height

The IK solver handles all the trigonometry for femur and knee angles
automatically, so we only need to think about WHERE we want each foot.

SETUP (same as v6)
-----
  ros2 run ros_gz_bridge parameter_bridge /imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import Imu
from builtin_interfaces.msg import Duration
import math
import time
import sys
import os

# Add the scripts directory to path so we can import leg_kinematics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leg_kinematics import leg_ik_to_command, LEGS, FEMUR_LENGTH, TIBIA_LENGTH


# ======================= TUNABLE PARAMETERS ========================
CYCLE_PERIOD = 0.6            # full trot cycle, seconds
CONTROL_RATE_HZ = 50.0        # Hz — trajectory update rate
SETTLE_TIME = 1.0             # seconds to hold standing pose before trotting
BLEND_DURATION = 2.0          # seconds to ramp gait amplitudes from 0→full

# --- Foot trajectory parameters (in meters, leg-local frame) ---
STRIDE_LENGTH = 0.04          # total fore/aft travel of the foot (meters)
STEP_HEIGHT   = 0.015         # how high the foot lifts during swing (meters)
STANCE_Z_HEIGHT = -0.220      # standing leg height (straighter legs = less torque needed)
BASE_HIP_SPRAWL = 0.025       # outward splay for wide stance (meters)
BODY_SHIFT_Y = 0.010          # shifts body left by 10mm to take weight off right legs
ROLL_OFFSET = 0.005           # Z-height offset: + extends right legs / shortens left legs to fix right-tilt

# --- IMU Yaw PD Controller Gains ---
YAW_KP = 0.1                 # proportional gain (lowered to prevent left/right weaving)
YAW_KD = 0.02                # derivative gain
YAW_MAX_CORRECTION = 0.15    # max differential steering clamp
# ===================================================================


# Diagonal pair phase offsets (same as before)
TROT_OFFSETS = {'FL': 0.0, 'BR': 0.0, 'FR': 0.5, 'BL': 0.5}

# Leg groups for differential steering
LEFT_LEGS  = {'FL', 'BL'}
RIGHT_LEGS = {'FR', 'BR'}


# ------------ Utility functions -------------------

def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

def smoother_step(t):
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

def soft_bell(frac):
    """sin² bell: smooth liftoff and touchdown."""
    return math.sin(math.pi * frac) ** 2

def quaternion_to_yaw(x, y, z, w):
    """Extract yaw from Z-up ENU world-frame quaternion (Gazebo convention)."""
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# ------------ Foot trajectory generator -------------------

def compute_foot_position(phase_01):
    """
    Compute the foot target position (x, y, z) in the leg-local frame
    for a given phase [0, 1).

    phase [0.0, 0.5) = SWING  — foot in the air, moving forward
    phase [0.5, 1.0) = STANCE — foot on the ground, pushing backward

    Returns (x, y, z) in meters, and is_swing boolean.

    Coordinate convention (leg-local):
      x = forward (+) / backward (-)
      y = outward (+) / inward (-)
      z = up (+) / down (-)
    """
    half_stride = STRIDE_LENGTH / 2.0

    if phase_01 < 0.5:
        # --- SWING PHASE ---
        frac = phase_01 / 0.5  # 0→1 over swing

        # X: sweep from rear (-half_stride) to front (+half_stride)
        x = -half_stride + STRIDE_LENGTH * smoother_step(frac)

        # Z: lift in a smooth bell curve, peak at mid-swing
        z = STANCE_Z_HEIGHT + STEP_HEIGHT * soft_bell(frac)

        is_swing = True
    else:
        # --- STANCE PHASE ---
        frac = (phase_01 - 0.5) / 0.5  # 0→1 over stance

        # X: push from front (+half_stride) to rear (-half_stride)
        x = half_stride - STRIDE_LENGTH * smoother_step(frac)

        # Z: dip slightly in the middle of stance to absorb bumping impact
        # This mimics the STANCE_HEIGHT_COMP from the hardcoded gait
        z = STANCE_Z_HEIGHT + 0.005 * math.sin(math.pi * frac)

        is_swing = False

    y = 0.0

    return x, y, z, is_swing


# -------------------------------------------------------------------

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

        # Joint order: all 12 actuated joints
        self.joint_order = [
            'Revolute 48', 'Revolute 49', 'Revolute 50', 'Revolute 51',
            'Revolute 13', 'Revolute 14', 'Revolute 15', 'Revolute 16',
            'Revolute 17', 'Revolute 18', 'Revolute 21', 'Revolute 22',
        ]

        self.blend_factor = 0.0

        # --- IMU State ---
        self.current_yaw = 0.0
        self.target_yaw = None
        self.prev_yaw_error = 0.0
        self.imu_received = False
        self.yaw_correction = 0.0

        self.get_logger().info(
            f"Trot gait v7 (IK + IMU) started: cycle={CYCLE_PERIOD}s, "
            f"stride={STRIDE_LENGTH}m, step_height={STEP_HEIGHT}m, "
            f"stance_z={STANCE_Z_HEIGHT}m, "
            f"YAW PD: Kp={YAW_KP}, Kd={YAW_KD}"
        )
        self.get_logger().info("Waiting for IMU data on /imu/data ...")

    def imu_callback(self, msg: Imu):
        q = msg.orientation
        self.current_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        if not self.imu_received:
            self.imu_received = True
            self.get_logger().info(
                f"IMU data received! Initial yaw = {math.degrees(self.current_yaw):.1f}°"
            )

    def tick(self):
        t = time.time() - self.start_time

        # ---- Phase 0: Settling (hold standing pose via IK) ----
        if t < SETTLE_TIME:
            self.publish_standing_pose()
            return

        # ---- Phase 1: Blend into gait ----
        if not self.gait_started:
            self.gait_started = True
            self.gait_start_time = time.time()
            self.target_yaw = self.current_yaw
            self.get_logger().info(
                f"Settling complete — starting IK trot. "
                f"Target heading locked: {math.degrees(self.target_yaw):.1f}°"
            )

        gait_t = time.time() - self.gait_start_time
        self.blend_factor = smoothstep(min(1.0, gait_t / BLEND_DURATION))

        # ---- IMU PD Yaw Controller ----
        if self.imu_received and self.target_yaw is not None:
            yaw_error = normalize_angle(self.current_yaw - self.target_yaw)
            dt = 1.0 / CONTROL_RATE_HZ
            d_error = (yaw_error - self.prev_yaw_error) / dt
            correction = YAW_KP * yaw_error + YAW_KD * d_error
            self.prev_yaw_error = yaw_error
            correction = max(-YAW_MAX_CORRECTION, min(YAW_MAX_CORRECTION, correction))
            self.yaw_correction = correction
        else:
            self.yaw_correction = 0.0

        # ---- Phase 2: Compute IK trot gait ----
        global_phase = (gait_t % CYCLE_PERIOD) / CYCLE_PERIOD

        commands = {}
        for leg_name in LEGS:
            # Each leg's local phase with diagonal offset
            local_phase = (global_phase + TROT_OFFSETS[leg_name]) % 1.0

            # Get the desired foot position in leg-local frame
            x, y, z, is_swing = compute_foot_position(local_phase)
            
            # IMU Yaw Correction via Hip Angles (Lateral Shift)
            # Instead of changing stride length, we use the IMU to dynamically shift 
            # the body left/right to counter the crab-walking that causes the deviation.
            dynamic_body_shift = BODY_SHIFT_Y + (self.yaw_correction * 0.1)

            # Apply sprawl and dynamic anti-crab body shift
            if leg_name in LEFT_LEGS:
                # Left legs: Y is outward (left). Shift body right by moving feet left.
                y += BASE_HIP_SPRAWL - dynamic_body_shift
            else:
                # Right legs: Y is outward (right). Shift body right by moving feet left.
                y += BASE_HIP_SPRAWL + dynamic_body_shift

            # Blend: interpolate between standing position and gait position
            x_blended = x * self.blend_factor
            y_blended = y * self.blend_factor
            
            # Apply Roll offset (right legs reach deeper to push body up)
            z_target = STANCE_Z_HEIGHT
            if leg_name in RIGHT_LEGS:
                z_target -= ROLL_OFFSET
            else:
                z_target += ROLL_OFFSET
                
            z_blended = STANCE_Z_HEIGHT + (z - STANCE_Z_HEIGHT) * self.blend_factor
            if not is_swing:
                # Apply roll extension when firmly planted
                z_blended = z_target + (z - STANCE_Z_HEIGHT) * self.blend_factor
            # Run inverse kinematics → joint angles
            try:
                leg_commands = leg_ik_to_command(leg_name, x_blended, y_blended, z_blended)
                commands.update(leg_commands)
            except ValueError as e:
                # Target unreachable — hold last position
                self.get_logger().warn(
                    f"IK unreachable for {leg_name}: {e} "
                    f"(x={x_blended:.4f}, y={y_blended:.4f}, z={z_blended:.4f})",
                    throttle_duration_sec=2.0
                )
                continue

        self.publish_commands(commands)

    def publish_standing_pose(self):
        """Send the IK-computed standing pose (feet directly below hips)."""
        commands = {}
        for leg_name in LEGS:
            y = 0.0
            if leg_name in LEFT_LEGS:
                y += BASE_HIP_SPRAWL - BODY_SHIFT_Y
            else:
                y += BASE_HIP_SPRAWL + BODY_SHIFT_Y
                
            try:
                leg_commands = leg_ik_to_command(leg_name, 0.0, y, STANCE_Z_HEIGHT)
                commands.update(leg_commands)
            except ValueError:
                pass
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