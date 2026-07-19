#!/usr/bin/env python3
"""
trot_gait.py — Combined forward + rotation state machine

Sequences the robot through a demo path:
  1. Walk FORWARD for FORWARD_DURATION seconds
  2. Turn LEFT 90°  (IMU-measured)
  3. Walk FORWARD for FORWARD_DURATION seconds
  4. Turn RIGHT 90° (IMU-measured)
  5. Walk FORWARD for FORWARD_DURATION seconds
  6. Stop (standing pose)

Runs until the sequence completes or Ctrl+C is pressed.

SETUP:
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
SETTLE_TIME = 1.0             # seconds to hold standing pose at start
BLEND_DURATION = 2.0          # seconds to ramp gait amplitudes from 0→full

# --- Forward gait parameters ---
STRIDE_LENGTH = 0.04          # fore/aft travel of foot (meters)
STEP_HEIGHT   = 0.015         # foot lift height during swing (meters)
STANCE_Z_HEIGHT = -0.220      # standing leg height (straighter legs = less torque needed)
BASE_HIP_SPRAWL = 0.025       # outward splay for wide stance (meters)
BODY_SHIFT_Y = 0.010          # shifts body left by 10mm to take weight off right legs
ROLL_OFFSET = 0.005           # Z-height offset: + extends right legs / shortens left legs to fix right-tilt

# --- Forward IMU Yaw PD Controller ---
YAW_KP = 0.1                 # proportional gain
YAW_KD = 0.02                # derivative gain
YAW_MAX_CORRECTION = 0.15    # max differential steering clamp

# --- Rotation gait parameters ---
ROTATION_STRIDE = 0.08        # how far each foot moves per step along its arc

# --- State machine timing ---
FORWARD_DURATION = 5.0        # seconds of forward walking per segment
TURN_ANGLE = math.radians(90) # target turn angle (90°)
TURN_TOLERANCE = math.radians(5)  # how close to target before declaring turn complete
TRANSITION_PAUSE = 0.5        # seconds to pause (standing) between state transitions
# ===================================================================


# Diagonal pair phase offsets
TROT_OFFSETS = {'FL': 0.0, 'BR': 0.0, 'FR': 0.5, 'BL': 0.5}

# Leg groups
LEFT_LEGS  = {'FL', 'BL'}
RIGHT_LEGS = {'FR', 'BR'}

# Hip positions relative to robot center (body frame: X=forward, Y=left)
HIP_BODY_POS = {
    'FL': ( 0.082,  0.043),
    'FR': ( 0.080, -0.043),
    'BL': (-0.080,  0.043),
    'BR': (-0.082, -0.043),
}

# State machine states
STATE_SETTLE     = 'SETTLE'
STATE_FORWARD_1  = 'FORWARD_1'
STATE_PAUSE_1    = 'PAUSE_1'
STATE_TURN_LEFT  = 'TURN_LEFT'
STATE_PAUSE_2    = 'PAUSE_2'
STATE_FORWARD_2  = 'FORWARD_2'
STATE_PAUSE_3    = 'PAUSE_3'
STATE_TURN_RIGHT = 'TURN_RIGHT'
STATE_PAUSE_4    = 'PAUSE_4'
STATE_FORWARD_3  = 'FORWARD_3'
STATE_DONE       = 'DONE'


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
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# ------------ Unicycle model: per-leg tangent directions ---

def compute_leg_tangents(turn_sign):
    """
    Compute normalized stride direction for each leg in leg-local frame.
    turn_sign: +1 for left (CCW), -1 for right (CW).
    """
    tangents = {}
    for leg, (rx, ry) in HIP_BODY_POS.items():
        tang_x_body = -turn_sign * ry
        tang_y_body = turn_sign * rx
        tx = tang_x_body
        if leg in LEFT_LEGS:
            ty = tang_y_body
        else:
            ty = -tang_y_body
        mag = math.sqrt(tx**2 + ty**2)
        tangents[leg] = (tx / mag, ty / mag)
    return tangents


# Precompute tangent directions for both turn directions
TANGENTS_LEFT  = compute_leg_tangents(+1)
TANGENTS_RIGHT = compute_leg_tangents(-1)


# ------------ Foot trajectory generators -------------------

def compute_forward_foot(phase_01):
    """Forward gait foot trajectory."""
    half_stride = STRIDE_LENGTH / 2.0

    if phase_01 < 0.5:
        frac = phase_01 / 0.5
        x = -half_stride + STRIDE_LENGTH * smoother_step(frac)
        z = STANCE_Z_HEIGHT + STEP_HEIGHT * soft_bell(frac)
        is_swing = True
    else:
        frac = (phase_01 - 0.5) / 0.5
        x = half_stride - STRIDE_LENGTH * smoother_step(frac)
        z = STANCE_Z_HEIGHT + 0.005 * math.sin(math.pi * frac)
        is_swing = False

    return x, 0.0, z, is_swing


def compute_rotation_foot(phase_01, tang_x, tang_y):
    """Rotation gait foot trajectory along tangent arc."""
    half_stride = ROTATION_STRIDE / 2.0

    if phase_01 < 0.5:
        frac = phase_01 / 0.5
        progress = -1.0 + 2.0 * smoother_step(frac)
        x = tang_x * half_stride * progress
        y = tang_y * half_stride * progress
        z = STANCE_Z_HEIGHT + STEP_HEIGHT * soft_bell(frac)
        is_swing = True
    else:
        frac = (phase_01 - 0.5) / 0.5
        progress = 1.0 - 2.0 * smoother_step(frac)
        x = tang_x * half_stride * progress
        y = tang_y * half_stride * progress
        z = STANCE_Z_HEIGHT + 0.005 * math.sin(math.pi * frac)
        is_swing = False

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
            Imu, '/imu/data', self.imu_callback, 10
        )

        self.start_time = time.time()
        self.timer = self.create_timer(1.0 / CONTROL_RATE_HZ, self.tick)

        # Joint order
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

        # --- State Machine ---
        self.state = STATE_SETTLE
        self.state_start_time = time.time()
        self.gait_time_offset = 0.0   # continuous gait clock for smooth phase
        self.turn_start_yaw = 0.0     # yaw at the start of a turn

        self.get_logger().info(
            f"Combined trot gait started: forward={FORWARD_DURATION}s segments, "
            f"turns=90°, stride={STRIDE_LENGTH}m, rotation_stride={ROTATION_STRIDE}m"
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

    def transition_to(self, new_state):
        """Transition to a new state, resetting the state timer."""
        self.get_logger().info(f"State: {self.state} → {new_state}")
        self.state = new_state
        self.state_start_time = time.time()
        self.prev_yaw_error = 0.0

        # When entering a FORWARD state, lock the current yaw as the heading target
        if new_state.startswith('FORWARD'):
            self.target_yaw = self.current_yaw
            self.get_logger().info(
                f"  Heading locked: {math.degrees(self.target_yaw):.1f}°"
            )

        # When entering a TURN state, record the starting yaw
        if new_state.startswith('TURN'):
            self.turn_start_yaw = self.current_yaw
            self.get_logger().info(
                f"  Turn starting from yaw: {math.degrees(self.turn_start_yaw):.1f}°"
            )

    def state_elapsed(self):
        """Time elapsed since entering the current state."""
        return time.time() - self.state_start_time

    def tick(self):
        t = time.time() - self.start_time

        # ======== STATE MACHINE ========

        if self.state == STATE_SETTLE:
            self.publish_standing_pose()
            if t >= SETTLE_TIME:
                if not self.imu_received:
                    self.get_logger().error("CRITICAL: NO IMU DATA RECEIVED! The IMU bridge is not running.")
                    self.get_logger().error("Turns will use a time-based fallback instead of measuring 90 degrees.")
                self.transition_to(STATE_FORWARD_1)
            return

        elif self.state == STATE_FORWARD_1:
            self.tick_forward()
            if self.state_elapsed() >= FORWARD_DURATION:
                self.transition_to(STATE_PAUSE_1)

        elif self.state == STATE_PAUSE_1:
            self.publish_standing_pose()
            if self.state_elapsed() >= TRANSITION_PAUSE:
                self.transition_to(STATE_TURN_LEFT)

        elif self.state == STATE_TURN_LEFT:
            self.tick_rotate(+1, TANGENTS_LEFT)
            turned = normalize_angle(self.current_yaw - self.turn_start_yaw)
            if abs(turned) >= TURN_ANGLE - TURN_TOLERANCE or self.state_elapsed() > 8.0:
                self.get_logger().info(
                    f"  Left turn complete! Turned {math.degrees(turned):.1f}°"
                )
                self.transition_to(STATE_PAUSE_2)

        elif self.state == STATE_PAUSE_2:
            self.publish_standing_pose()
            if self.state_elapsed() >= TRANSITION_PAUSE:
                self.transition_to(STATE_FORWARD_2)

        elif self.state == STATE_FORWARD_2:
            self.tick_forward()
            if self.state_elapsed() >= FORWARD_DURATION:
                self.transition_to(STATE_PAUSE_3)

        elif self.state == STATE_PAUSE_3:
            self.publish_standing_pose()
            if self.state_elapsed() >= TRANSITION_PAUSE:
                self.transition_to(STATE_TURN_RIGHT)

        elif self.state == STATE_TURN_RIGHT:
            self.tick_rotate(-1, TANGENTS_RIGHT)
            turned = normalize_angle(self.current_yaw - self.turn_start_yaw)
            if abs(turned) >= TURN_ANGLE - TURN_TOLERANCE or self.state_elapsed() > 8.0:
                self.get_logger().info(
                    f"  Right turn complete! Turned {math.degrees(turned):.1f}°"
                )
                self.transition_to(STATE_PAUSE_4)

        elif self.state == STATE_PAUSE_4:
            self.publish_standing_pose()
            if self.state_elapsed() >= TRANSITION_PAUSE:
                self.transition_to(STATE_FORWARD_3)

        elif self.state == STATE_FORWARD_3:
            self.tick_forward()
            if self.state_elapsed() >= FORWARD_DURATION:
                self.transition_to(STATE_DONE)

        elif self.state == STATE_DONE:
            self.publish_standing_pose()

    # ---- Forward walking tick ----

    def tick_forward(self):
        gait_t = self.state_elapsed()
        self.blend_factor = smoothstep(min(1.0, gait_t / BLEND_DURATION))

        # IMU PD yaw controller (keep heading straight)
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

        global_phase = (gait_t % CYCLE_PERIOD) / CYCLE_PERIOD

        commands = {}
        for leg_name in LEGS:
            local_phase = (global_phase + TROT_OFFSETS[leg_name]) % 1.0
            x, y, z, is_swing = compute_forward_foot(local_phase)

            # IMU yaw correction via hip lateral shift
            dynamic_body_shift = BODY_SHIFT_Y + (self.yaw_correction * 0.1)
            if leg_name in LEFT_LEGS:
                y += BASE_HIP_SPRAWL - dynamic_body_shift
            else:
                y += BASE_HIP_SPRAWL + dynamic_body_shift

            # Blend
            x_b = x * self.blend_factor
            y_b = y * self.blend_factor
            z_b = STANCE_Z_HEIGHT + (z - STANCE_Z_HEIGHT) * self.blend_factor

            try:
                leg_commands = leg_ik_to_command(leg_name, x_b, y_b, z_b)
                commands.update(leg_commands)
            except ValueError as e:
                self.get_logger().warn(
                    f"IK unreachable for {leg_name}: {e}",
                    throttle_duration_sec=2.0
                )
                continue

        self.publish_commands(commands)

    # ---- Rotation tick ----

    def tick_rotate(self, turn_sign, tangents):
        gait_t = self.state_elapsed()
        self.blend_factor = smoothstep(min(1.0, gait_t / BLEND_DURATION))

        global_phase = (gait_t % CYCLE_PERIOD) / CYCLE_PERIOD

        commands = {}
        for leg_name in LEGS:
            local_phase = (global_phase + TROT_OFFSETS[leg_name]) % 1.0
            tang_x, tang_y = tangents[leg_name]
            x, y, z, is_swing = compute_rotation_foot(local_phase, tang_x, tang_y)

            # Add sprawl
            y += BASE_HIP_SPRAWL

            # Blend
            x_b = x * self.blend_factor
            y_b = y * self.blend_factor
            z_b = STANCE_Z_HEIGHT + (z - STANCE_Z_HEIGHT) * self.blend_factor

            try:
                leg_commands = leg_ik_to_command(leg_name, x_b, y_b, z_b)
                commands.update(leg_commands)
            except ValueError as e:
                self.get_logger().warn(
                    f"IK unreachable for {leg_name}: {e}",
                    throttle_duration_sec=2.0
                )
                continue

        self.publish_commands(commands)

        # Log yaw progress
        if self.imu_received:
            turned = normalize_angle(self.current_yaw - self.turn_start_yaw)
            direction = "LEFT" if turn_sign > 0 else "RIGHT"
            self.get_logger().info(
                f"  Turning {direction}: {math.degrees(turned):.1f}° / "
                f"{math.degrees(TURN_ANGLE):.0f}° target",
                throttle_duration_sec=1.0
            )

    # ---- Shared helpers ----

    def publish_standing_pose(self):
        commands = {}
        for leg_name in LEGS:
            y = BASE_HIP_SPRAWL
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
