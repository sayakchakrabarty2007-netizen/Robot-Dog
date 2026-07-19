#!/usr/bin/env python3
"""
trot_gait_rotate.py — In-place yaw rotation using IK + unicycle model

Makes the robot rotate left or right in place (pure yaw, no forward motion).
Each leg's stride direction is computed from the unicycle turning model:
the foot traces an arc tangent to a circle centered at the robot's
geometric center. This naturally makes each leg step in the correct
direction and magnitude for pure rotation.

Usage:
    python3 trot_gait_rotate.py --direction left
    python3 trot_gait_rotate.py --direction right

Rotates continuously until Ctrl+C.

SETUP (same as trot_gait_forward.py):
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
import argparse

# Add the scripts directory to path so we can import leg_kinematics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leg_kinematics import leg_ik_to_command, LEGS, FEMUR_LENGTH, TIBIA_LENGTH


# ======================= TUNABLE PARAMETERS ========================
CYCLE_PERIOD = 0.6            # full trot cycle, seconds
CONTROL_RATE_HZ = 50.0        # Hz — trajectory update rate
SETTLE_TIME = 1.0             # seconds to hold standing pose before rotating
BLEND_DURATION = 2.0          # seconds to ramp gait amplitudes from 0→full

# --- Foot trajectory parameters (in meters, leg-local frame) ---
ROTATION_STRIDE = 0.08        # how far each foot moves per step along its arc (meters)
                               # Increase → faster rotation. Decrease → slower, more stable.
STEP_HEIGHT   = 0.015         # foot lift height during swing (meters)
STANCE_Z_HEIGHT = -0.170      # MUST be bent! Max leg length is 0.227m
BASE_HIP_SPRAWL = 0.025       # outward splay for wide stance (meters)
PITCH_OFFSET = 0.005          # tilt compensation: shortens front legs, lengthens back legs (meters)
# ===================================================================


# Diagonal pair phase offsets (same as forward trot)
TROT_OFFSETS = {'FL': 0.0, 'BR': 0.0, 'FR': 0.5, 'BL': 0.5}

# Leg groups
LEFT_LEGS  = {'FL', 'BL'}
RIGHT_LEGS = {'FR', 'BR'}

# Hip positions relative to robot center (body frame: X=forward, Y=left)
# Computed from URDF: base_link → Hip_servos_holder → Revolute (hip joint)
HIP_BODY_POS = {
    'FL': ( 0.082,  0.043),   # front-left
    'FR': ( 0.080, -0.043),   # front-right
    'BL': (-0.080,  0.043),   # back-left
    'BR': (-0.082, -0.043),   # back-right
}


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


# ------------ Unicycle model: compute per-leg stride directions ---

def compute_leg_tangents(turn_sign):
    """
    Compute the normalized stride direction for each leg in leg-local frame.

    For pure rotation, each foot traces an arc tangent to a circle centered
    at the robot's geometric center. The tangent direction at each hip
    determines which way that leg needs to stride.

    Args:
        turn_sign: +1 for left turn (CCW from top), -1 for right turn (CW).

    Returns:
        dict {leg_name: (tang_x, tang_y)} — unit tangent direction per leg,
        expressed in the leg-local frame (x=forward, y=outward).
    """
    tangents = {}
    for leg, (rx, ry) in HIP_BODY_POS.items():
        # Body-frame tangent for rotation around center
        # For CCW rotation: tangent = (-ry, rx)
        tang_x_body = -turn_sign * ry
        tang_y_body = turn_sign * rx

        # Convert body-frame Y to leg-local Y (outward)
        # Left legs: outward = left = +Y_body
        # Right legs: outward = right = -Y_body
        tx = tang_x_body
        if leg in LEFT_LEGS:
            ty = tang_y_body
        else:
            ty = -tang_y_body

        # Normalize to unit vector
        mag = math.sqrt(tx**2 + ty**2)
        tangents[leg] = (tx / mag, ty / mag)

    return tangents


# ------------ Foot trajectory generator -------------------

def compute_rotation_foot_position(phase_01, tang_x, tang_y):
    """
    Compute the foot target position (x, y, z) in leg-local frame for
    a rotation gait. The stride is along (tang_x, tang_y) instead of
    pure X (forward).

    phase [0.0, 0.5) = SWING  — foot in the air, repositioning along arc
    phase [0.5, 1.0) = STANCE — foot on ground, pushing to rotate body

    Returns (x, y, z, is_swing).
    """
    half_stride = ROTATION_STRIDE / 2.0

    if phase_01 < 0.5:
        # --- SWING PHASE ---
        # Reposition foot: sweep from rear of arc to front of arc, lifting up
        frac = phase_01 / 0.5  # 0→1 over swing
        progress = -1.0 + 2.0 * smoother_step(frac)  # -1 → +1 smoothly

        x = tang_x * half_stride * progress
        y = tang_y * half_stride * progress
        z = STANCE_Z_HEIGHT + STEP_HEIGHT * soft_bell(frac)
        is_swing = True
    else:
        # --- STANCE PHASE ---
        # Foot planted on ground, body rotates over it.
        # From body's perspective, foot slides from front to rear of arc.
        frac = (phase_01 - 0.5) / 0.5  # 0→1 over stance
        progress = 1.0 - 2.0 * smoother_step(frac)  # +1 → -1 smoothly

        x = tang_x * half_stride * progress
        y = tang_y * half_stride * progress
        # Z-dip: gentle compression mid-stance for suspension
        z = STANCE_Z_HEIGHT + 0.005 * math.sin(math.pi * frac)
        is_swing = False

    return x, y, z, is_swing


# -------------------------------------------------------------------

class RotateGaitNode(Node):
    def __init__(self, turn_sign):
        super().__init__('rotate_gait_node')

        self.turn_sign = turn_sign
        self.direction_name = 'LEFT' if turn_sign > 0 else 'RIGHT'

        # Precompute per-leg tangent directions
        self.leg_tangents = compute_leg_tangents(turn_sign)

        # --- Publishers ---
        self.publisher_ = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # --- IMU Subscriber (for yaw tracking/logging only) ---
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

        # --- IMU State (tracking only, no correction) ---
        self.current_yaw = 0.0
        self.initial_yaw = None
        self.imu_received = False

        self.get_logger().info(
            f"Rotate gait ({self.direction_name}) started: cycle={CYCLE_PERIOD}s, "
            f"rotation_stride={ROTATION_STRIDE}m, step_height={STEP_HEIGHT}m, "
            f"stance_z={STANCE_Z_HEIGHT}m"
        )
        self.get_logger().info(
            f"Per-leg tangent directions (leg-local): "
            + ", ".join(f"{leg}=({tx:.2f},{ty:.2f})" for leg, (tx, ty) in self.leg_tangents.items())
        )
        self.get_logger().info("Waiting for IMU data on /imu/data ...")

    def imu_callback(self, msg: Imu):
        q = msg.orientation
        self.current_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        if not self.imu_received:
            self.imu_received = True
            self.initial_yaw = self.current_yaw
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
            self.get_logger().info(
                f"Settling complete — starting {self.direction_name} rotation."
            )

        gait_t = time.time() - self.gait_start_time
        self.blend_factor = smoothstep(min(1.0, gait_t / BLEND_DURATION))

        # ---- Log yaw progress every 2 seconds ----
        if self.imu_received and self.initial_yaw is not None:
            total_rotation = normalize_angle(self.current_yaw - self.initial_yaw)
            if int(gait_t * 10) % 20 == 0:  # roughly every 2s
                self.get_logger().info(
                    f"Yaw: {math.degrees(self.current_yaw):.1f}° "
                    f"(rotated {math.degrees(total_rotation):.1f}° from start)",
                    throttle_duration_sec=2.0
                )

        # ---- Phase 2: Compute rotation gait ----
        global_phase = (gait_t % CYCLE_PERIOD) / CYCLE_PERIOD

        commands = {}
        for leg_name in LEGS:
            # Each leg's local phase with diagonal offset
            local_phase = (global_phase + TROT_OFFSETS[leg_name]) % 1.0

            # Get this leg's tangent direction
            tang_x, tang_y = self.leg_tangents[leg_name]

            # Compute foot position along the tangent arc
            x, y, z, is_swing = compute_rotation_foot_position(local_phase, tang_x, tang_y)

            # Add outward sprawl for stability
            y += BASE_HIP_SPRAWL

            # Pitch compensation
            z_target = STANCE_Z_HEIGHT
            if leg_name in ['FL', 'FR']:
                z_target += PITCH_OFFSET
            else:
                z_target -= PITCH_OFFSET

            # Blend: interpolate between standing position and gait position
            # Shift X backwards by 2.5cm to align support polygon with CoM
            x_blended = -0.025 + x * self.blend_factor
            y_blended = y * self.blend_factor
            z_blended = z_target + (z - STANCE_Z_HEIGHT) * self.blend_factor

            # Run inverse kinematics → joint angles
            try:
                leg_commands = leg_ik_to_command(leg_name, x_blended, y_blended, z_blended)
                commands.update(leg_commands)
            except ValueError as e:
                self.get_logger().warn(
                    f"IK unreachable for {leg_name}: {e} "
                    f"(x={x_blended:.4f}, y={y_blended:.4f}, z={z_blended:.4f})",
                    throttle_duration_sec=2.0
                )
                continue

        self.publish_commands(commands)

    def publish_standing_pose(self):
        """Send the IK-computed standing pose (feet directly below hips with sprawl)."""
        commands = {}
        for leg_name in LEGS:
            z_stand = STANCE_Z_HEIGHT
            if leg_name in ['FL', 'FR']:
                z_stand += PITCH_OFFSET
            else:
                z_stand -= PITCH_OFFSET

            try:
                # Shift X backwards by 2.5cm to align support polygon with CoM
                leg_commands = leg_ik_to_command(leg_name, -0.025, BASE_HIP_SPRAWL, z_stand)
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
    parser = argparse.ArgumentParser(description='In-place rotation gait for quadruped robot')
    parser.add_argument(
        '--direction', type=str, required=True, choices=['left', 'right'],
        help='Rotation direction: "left" (CCW from top) or "right" (CW from top)'
    )
    parsed_args = parser.parse_args()

    turn_sign = +1 if parsed_args.direction == 'left' else -1

    rclpy.init()
    node = RotateGaitNode(turn_sign)
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
