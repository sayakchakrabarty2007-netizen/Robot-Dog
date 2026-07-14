#!/usr/bin/env python3
"""
Trot Gait Controller for 12-DOF Quadruped Robot
================================================
Phase 2: Inverse Kinematics (geometric 2-link IK)
Phase 3: Gait Generator (trot pattern with body pitch compensation)

Usage:
  # Terminal 1 — Launch Gazebo:
  ros2 launch Robot_description gazebo.launch.py

  # Terminal 2 — Stand first, then trot:
  python3 trot_gait.py --stand          # hold standing pose
  python3 trot_gait.py --trot           # start trotting
  python3 trot_gait.py --trot-in-place  # trot without forward movement
"""
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math
import sys
import time

# ================================================================
# SECTION 1: ROBOT DIMENSIONS (measured from URDF joint origins)
# ================================================================
# Femur link: from hip-pitch pivot to knee pivot
#   URDF Rev 17 origin relative to femur parent: (0.054898, -0.094919, -0.002)
#   Effective 2D length in sagittal plane:
FEMUR_LENGTH = math.sqrt(0.054898**2 + 0.094919**2)  # = 0.1097 m

# Tibia link: from knee pivot to foot contact
#   Estimated from tibia CoM position and visual extents
TIBIA_LENGTH = 0.100  # = 0.100 m

# At URDF joint angle 0, the femur already tilts forward from vertical by:
#   atan2(forward_offset, downward_offset) = atan2(0.055, 0.095)
FEMUR_REST_ANGLE = math.atan2(0.054898, 0.094919)  # ≈ 0.524 rad (30°)


# ================================================================
# SECTION 2: GAIT PARAMETERS (all tunable)
# ================================================================
# Standing foot position relative to hip (computed from FK at angle 0,0)
STAND_X = FEMUR_LENGTH * math.sin(FEMUR_REST_ANGLE) + \
          TIBIA_LENGTH * math.sin(FEMUR_REST_ANGLE)  # ≈ 0.105 m forward
STAND_Z = FEMUR_LENGTH * math.cos(FEMUR_REST_ANGLE) + \
          TIBIA_LENGTH * math.cos(FEMUR_REST_ANGLE)  # ≈ 0.182 m down

# Gait geometry
STEP_HEIGHT  = 0.025    # 25 mm foot lift during swing phase
STEP_LENGTH  = 0.035    # 35 mm forward per step (total stride = 2x)
CYCLE_TIME   = 1.0      # 1.0 second per full gait cycle
PITCH_COMP   = 0.010    # 10 mm backward body lean for front-leg stability
RAMP_CYCLES  = 3        # Gradually increase step height over first N cycles

# Plate-to-knee coupling ratio (1.0 = plate follows knee 1:1)
PLATE_COUPLING = 1.0


# ================================================================
# SECTION 3: JOINT MAP (verified by user testing on 2026-07-14)
# ================================================================
# Joint order must match controllers.yaml
ALL_JOINT_NAMES = [
    'Revolute 48', 'Revolute 49', 'Revolute 50', 'Revolute 51',  # hips
    'Revolute 13', 'Revolute 14', 'Revolute 15', 'Revolute 16',  # femurs
    'Revolute 17', 'Revolute 18', 'Revolute 21', 'Revolute 22',  # knees
    'Revolute 24', 'Revolute 25', 'Revolute 26', 'Revolute 27',  # plates
]

# Per-leg configuration: (joint_name, sign_multiplier)
# sign_multiplier converts canonical direction → URDF direction
# Canonical: +forward for femur/knee, +outward for hip
#
# Verified directions:
#   Rev 48 BR hip:   + inside  - outside  → sign = -1 (canonical outward = URDF -)
#   Rev 49 BL hip:   - inside  + outside  → sign = +1
#   Rev 50 FL hip:   - inside  + outside  → sign = +1
#   Rev 51 FR hip:   + inside  - outside  → sign = -1
#   Rev 13 FL femur: + forward - backward → sign = +1
#   Rev 14 BL femur: + forward - backward → sign = +1
#   Rev 15 FR femur: - forward + backward → sign = -1
#   Rev 16 BR femur: - forward + backward → sign = -1
#   Rev 17 FR knee:  - forward + backward → sign = -1
#   Rev 18 BR knee:  - forward + backward → sign = -1
#   Rev 21 FL knee:  + forward - backward → sign = +1
#   Rev 22 BL knee:  + forward - backward → sign = +1
#   Rev 24 BL plate: - forward + backward → sign = -1
#   Rev 25 BR plate: - forward + backward → sign = -1
#   Rev 26 FR plate: - forward + backward → sign = -1
#   Rev 27 FL plate: + forward - backward → sign = +1

LEG_CONFIG = {
    #        hip_joint,       hip_sign, femur_joint,     fem_sign, knee_joint,      knee_sign, plate_joint,     plate_sign
    'FR': (('Revolute 51',   -1),      ('Revolute 15',  -1),      ('Revolute 17',  -1),       ('Revolute 26',  -1)),
    'FL': (('Revolute 50',   +1),      ('Revolute 13',  +1),      ('Revolute 21',  +1),       ('Revolute 27',  +1)),
    'BR': (('Revolute 48',   -1),      ('Revolute 16',  -1),      ('Revolute 18',  -1),       ('Revolute 25',  -1)),
    'BL': (('Revolute 49',   +1),      ('Revolute 14',  +1),      ('Revolute 22',  +1),       ('Revolute 24',  -1)),
}


# ================================================================
# SECTION 4: INVERSE KINEMATICS (2-link geometric)
# ================================================================
def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def ik_leg(foot_x, foot_z):
    """
    Compute hip-pitch (femur) and knee angles for a desired foot position.

    Coordinate frame (per leg, relative to hip-pitch pivot):
        +x = forward       (the direction the robot walks)
        +z = downward       (toward the ground)

    At URDF angle (0, 0), the foot is at (STAND_X, STAND_Z).

    Args:
        foot_x: desired foot forward offset from hip (m)
        foot_z: desired foot downward offset from hip (m, positive = lower)

    Returns:
        (femur_angle, knee_angle) in CANONICAL convention:
            femur_angle: + = swing leg forward from rest
            knee_angle:  0 = straight (aligned with femur), - = bend backward
    """
    L1, L2 = FEMUR_LENGTH, TIBIA_LENGTH

    # Distance from hip to desired foot position
    d = math.sqrt(foot_x**2 + foot_z**2)
    d = clamp(d, abs(L1 - L2) + 0.001, L1 + L2 - 0.001)  # stay in workspace

    # --- Knee angle (interior angle via law of cosines) ---
    cos_beta = (L1**2 + L2**2 - d**2) / (2.0 * L1 * L2)
    cos_beta = clamp(cos_beta, -1.0, 1.0)
    beta = math.acos(cos_beta)          # interior angle at knee joint
    knee_angle = -(math.pi - beta)      # 0 = straight, negative = bent backward

    # --- Femur angle (law of cosines at hip) ---
    cos_alpha = (L1**2 + d**2 - L2**2) / (2.0 * L1 * d)
    cos_alpha = clamp(cos_alpha, -1.0, 1.0)
    alpha = math.acos(cos_alpha)        # angle at hip between femur and line-to-foot

    gamma = math.atan2(foot_x, foot_z)  # angle from vertical to foot target

    # Knee-forward configuration (dog-leg): femur is ABOVE the line to foot
    phi = gamma + alpha                 # absolute femur angle from vertical
    femur_angle = phi - FEMUR_REST_ANGLE  # relative to rest orientation

    return femur_angle, knee_angle


def fk_leg(femur_angle, knee_angle):
    """
    Forward kinematics: joint angles → foot position.
    Useful for verification.
    """
    L1, L2 = FEMUR_LENGTH, TIBIA_LENGTH
    phi = FEMUR_REST_ANGLE + femur_angle   # absolute femur angle
    psi = phi + knee_angle                 # absolute tibia angle

    foot_x = L1 * math.sin(phi) + L2 * math.sin(psi)
    foot_z = L1 * math.cos(phi) + L2 * math.cos(psi)
    return foot_x, foot_z


# ================================================================
# SECTION 5: GAIT TRAJECTORY GENERATOR
# ================================================================
class TrotGait:
    """
    Trot gait: diagonal leg pairs move together.

    Full cycle (phase 0→1):
        Phase A (0.0 – 0.5): FR + BL swing;  FL + BR stance
        Phase B (0.5 – 1.0): FL + BR swing;  FR + BL stance

    Swing trajectory: semi-elliptical arc (foot lifts, advances, lowers)
    Stance trajectory: foot pushes backward on the ground
    """

    def __init__(self, step_height=STEP_HEIGHT, step_length=STEP_LENGTH,
                 stand_x=STAND_X, stand_z=STAND_Z, pitch_comp=PITCH_COMP):
        self.step_height = step_height
        self.step_length = step_length
        self.stand_x = stand_x
        self.stand_z = stand_z
        self.pitch_comp = pitch_comp

    def get_foot_targets(self, phase):
        """
        Returns {leg_name: (foot_x, foot_z)} for all 4 legs at given phase.
        phase: 0.0 → 1.0 (one full gait cycle)
        """
        targets = {}

        if phase < 0.5:
            # --- Phase A: FR+BL swing, FL+BR stance ---
            t = phase / 0.5                   # 0→1 within this half-cycle

            targets['FR'] = self._swing(t)
            targets['BL'] = self._swing(t)
            targets['FL'] = self._stance(t, is_front=True)
            targets['BR'] = self._stance(t, is_front=False)
        else:
            # --- Phase B: FL+BR swing, FR+BL stance ---
            t = (phase - 0.5) / 0.5

            targets['FL'] = self._swing(t)
            targets['BR'] = self._swing(t)
            targets['FR'] = self._stance(t, is_front=True)
            targets['BL'] = self._stance(t, is_front=False)

        return targets

    def _swing(self, t):
        """
        Swing trajectory: foot lifts from behind, arcs forward, places down in front.

        t=0: foot is at the BACK of the stride (just finished stance)
        t=1: foot is at the FRONT of the stride (about to touch down)

        Path: x linear, z sinusoidal lift
        """
        x = self.stand_x - self.step_length / 2.0 + self.step_length * t
        z = self.stand_z - self.step_height * math.sin(math.pi * t)
        return (x, z)

    def _stance(self, t, is_front=False):
        """
        Stance trajectory: foot stays on ground and pushes backward.

        t=0: foot is at the FRONT (just touched down)
        t=1: foot is at the BACK (about to lift off)

        Body pitch compensation: front stance legs shift forward to
        lean the body backward, preventing forward tipping when the
        diagonal front leg is in the air.
        """
        x = self.stand_x + self.step_length / 2.0 - self.step_length * t

        # Apply pitch compensation to the front stance leg
        if is_front:
            x += self.pitch_comp

        z = self.stand_z
        return (x, z)


# ================================================================
# SECTION 6: ROS 2 CONTROLLER NODE
# ================================================================
class DogController(Node):
    def __init__(self):
        super().__init__('dog_controller')

        self.pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Determine mode from command line
        self.mode = 'stand'
        if '--trot' in sys.argv:
            self.mode = 'trot'
        elif '--trot-in-place' in sys.argv:
            self.mode = 'trot-in-place'

        self.gait = TrotGait()
        if self.mode == 'trot-in-place':
            self.gait.step_length = 0.0  # no forward movement

        self.phase = 0.0
        self.cycle_count = 0
        self.dt = 0.02  # 50 Hz control loop

        self.get_logger().info(f"=== Dog Controller [{self.mode}] ===")
        self.get_logger().info(f"  Femur length:   {FEMUR_LENGTH*1000:.1f} mm")
        self.get_logger().info(f"  Tibia length:   {TIBIA_LENGTH*1000:.1f} mm")
        self.get_logger().info(f"  Standing foot:  ({STAND_X*1000:.1f}, {STAND_Z*1000:.1f}) mm")
        self.get_logger().info(f"  Step height:    {STEP_HEIGHT*1000:.1f} mm")
        self.get_logger().info(f"  Step length:    {self.gait.step_length*1000:.1f} mm")
        self.get_logger().info(f"  Cycle time:     {CYCLE_TIME:.2f} s")
        self.get_logger().info(f"  Pitch comp:     {PITCH_COMP*1000:.1f} mm")
        self.get_logger().info("Waiting 2s for controllers...")

        # Delay start to let Gazebo controllers initialize
        self.create_timer(2.0, self._on_ready, callback_group=None)
        self._started = False

    def _on_ready(self):
        if self._started:
            return
        self._started = True

        if self.mode == 'stand':
            self.get_logger().info("Sending STANDING pose (all joints = 0)...")
            self._send_all_zeros()
            self.get_logger().info("Done! Robot should be standing stiff.")
            self.get_logger().info("To trot:  python3 trot_gait.py --trot")
        else:
            self.get_logger().info("Sending standing pose first...")
            self._send_all_zeros()
            time.sleep(1.5)
            self.get_logger().info("Starting gait loop!")
            self.gait_timer = self.create_timer(self.dt, self._tick)

    # ---- Gait loop (called at 50 Hz) ----
    def _tick(self):
        # Ramp up step height over the first few cycles
        ramp = min(1.0, self.cycle_count / max(1, RAMP_CYCLES))
        self.gait.step_height = STEP_HEIGHT * ramp

        # Get foot targets from gait generator
        targets = self.gait.get_foot_targets(self.phase)

        # Build joint angle dictionary
        angles = {}
        for leg_name, (foot_x, foot_z) in targets.items():
            # --- IK ---
            femur_can, knee_can = ik_leg(foot_x, foot_z)

            # --- Map to URDF joints ---
            (hip_j, hip_s), (fem_j, fem_s), (knee_j, knee_s), (plate_j, plate_s) = LEG_CONFIG[leg_name]

            angles[hip_j]   = 0.0                              # no lateral roll for trot
            angles[fem_j]   = femur_can * fem_s                # femur
            angles[knee_j]  = knee_can  * knee_s               # knee (tibia)
            angles[plate_j] = knee_can  * PLATE_COUPLING * plate_s  # plate follows knee

        self._publish(angles)

        # Advance phase
        self.phase += self.dt / CYCLE_TIME
        if self.phase >= 1.0:
            self.phase -= 1.0
            self.cycle_count += 1
            if self.cycle_count <= RAMP_CYCLES:
                self.get_logger().info(
                    f"Cycle {self.cycle_count}: step_height ramping "
                    f"({self.gait.step_height*1000:.0f}/{STEP_HEIGHT*1000:.0f} mm)"
                )

    # ---- Helpers ----
    def _send_all_zeros(self):
        self._publish({name: 0.0 for name in ALL_JOINT_NAMES})

    def _publish(self, angles_dict):
        msg = JointTrajectory()
        msg.joint_names = list(ALL_JOINT_NAMES)

        pt = JointTrajectoryPoint()
        pt.positions = [angles_dict.get(n, 0.0) for n in ALL_JOINT_NAMES]
        pt.time_from_start = Duration(sec=0, nanosec=100_000_000)  # 100 ms
        msg.points.append(pt)

        self.pub.publish(msg)


# ================================================================
# SECTION 7: SELF-TEST & MAIN
# ================================================================
def self_test():
    """Verify IK/FK roundtrip before launching ROS."""
    print("=" * 60)
    print("  IK / FK Self-Test")
    print("=" * 60)

    # Test 1: Standing position should give (0, 0) angles
    f, k = ik_leg(STAND_X, STAND_Z)
    fx, fz = fk_leg(f, k)
    print(f"  Standing foot target: ({STAND_X*1000:.1f}, {STAND_Z*1000:.1f}) mm")
    print(f"  IK angles:  femur={math.degrees(f):+.2f}°  knee={math.degrees(k):+.2f}°")
    print(f"  FK verify:  ({fx*1000:.1f}, {fz*1000:.1f}) mm")
    assert abs(f) < 0.01 and abs(k) < 0.01, "Standing IK failed!"
    print("  ✓ Standing roundtrip OK\n")

    # Test 2: Swing peak (foot lifted and forward)
    sx = STAND_X + STEP_LENGTH / 2
    sz = STAND_Z - STEP_HEIGHT
    f, k = ik_leg(sx, sz)
    fx, fz = fk_leg(f, k)
    print(f"  Swing peak target:   ({sx*1000:.1f}, {sz*1000:.1f}) mm")
    print(f"  IK angles:  femur={math.degrees(f):+.2f}°  knee={math.degrees(k):+.2f}°")
    print(f"  FK verify:  ({fx*1000:.1f}, {fz*1000:.1f}) mm")
    err = math.sqrt((fx - sx)**2 + (fz - sz)**2)
    assert err < 0.001, f"Swing IK error too large: {err*1000:.2f} mm"
    print("  ✓ Swing roundtrip OK\n")

    # Test 3: Stance rear (foot pushed backward)
    bx = STAND_X - STEP_LENGTH / 2
    bz = STAND_Z
    f, k = ik_leg(bx, bz)
    fx, fz = fk_leg(f, k)
    print(f"  Stance rear target:  ({bx*1000:.1f}, {bz*1000:.1f}) mm")
    print(f"  IK angles:  femur={math.degrees(f):+.2f}°  knee={math.degrees(k):+.2f}°")
    print(f"  FK verify:  ({fx*1000:.1f}, {fz*1000:.1f}) mm")
    err = math.sqrt((fx - bx)**2 + (fz - bz)**2)
    assert err < 0.001, f"Stance IK error too large: {err*1000:.2f} mm"
    print("  ✓ Stance roundtrip OK\n")

    print("  All self-tests PASSED!")
    print("=" * 60)


def main():
    self_test()
    print()

    rclpy.init()
    node = DogController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
