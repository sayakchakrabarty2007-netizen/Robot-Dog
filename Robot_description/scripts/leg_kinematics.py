#!/usr/bin/env python3
"""
leg_kinematics.py

Inverse kinematics for a 3-DOF leg: hip (abduction/adduction), femur (pitch),
tibia/knee (pitch). Built from the actual joint-origin distances measured in
your URDF, and calibrated to the exact sign conventions you tested manually
with move_dog.py.

COORDINATE CONVENTION (leg-local frame, per leg):
  x = forward (+) / backward (-)
  y = outward, away from body (+) / inward, toward body (-)
  z = up (+) / down (-)   (foot target z is typically negative -- below hip)

This is a STANDARD coxa-femur-tibia leg model. femur_length and tibia_length
were computed from your URDF joint origins:
  femur_length = sqrt(0.054898^2 + 0.094919^2) = 0.1097 m
  tibia_length = sqrt(0.058442^2 + 0.101786^2) = 0.1174 m

TUNING NOTE:
  The sign conventions below (HIP_SIGN, FEMUR_SIGN, KNEE_SIGN) come directly
  from your manual joint testing. However, the *zero reference* assumption
  (e.g. whether femur=0 means "horizontal" or some other pose, and which
  direction is geometrically "down") could not be independently verified
  from the URDF alone. If a leg moves opposite to what you expect once you
  run this, flip the corresponding sign in FEMUR_SIGN / KNEE_SIGN / HIP_SIGN
  for that leg -- do NOT touch the IK math itself, just the sign table.
"""

import math

# ---- measured link lengths (meters) ----
FEMUR_LENGTH = 0.1097
TIBIA_LENGTH = 0.1174
COXA_LENGTH = 0.0    # hip-roll-axis to femur-pitch-axis offset is small in
                      # your design (~0.02m dominated by other axes); start
                      # at 0 and add a real measured value later if legs
                      # don't reach where expected.

# IK configuration
# The math yields 2 valid solutions (knee bends forward vs backward). 
# Change this between +1.0 and -1.0 to flip the knee bend direction.
KNEE_BEND_DIR = -1.0 

# ---- per-leg joint names (from your Robot.xacro) ----
LEGS = {
    'FL': {'hip': 'Revolute 50', 'femur': 'Revolute 13', 'knee': 'Revolute 21'},
    'FR': {'hip': 'Revolute 51', 'femur': 'Revolute 15', 'knee': 'Revolute 17'},
    'BL': {'hip': 'Revolute 49', 'femur': 'Revolute 14', 'knee': 'Revolute 22'},
    'BR': {'hip': 'Revolute 48', 'femur': 'Revolute 16', 'knee': 'Revolute 18'},
}

# ---- sign conventions from YOUR manual testing ----
# Positive canonical hip angle = leg abducts OUTWARD (away from body).
HIP_SIGN = {
    'FR': -1,   # 51: + = inside -> outside needs negative canonical->cmd... see note below
    'FL': +1,   # 50: - = inside, + = outside  -> matches canonical directly
    'BR': -1,   # 48: + = inside, - = outside  -> flip
    'BL': +1,   # 49: - = inside, + = outside  -> matches canonical directly
}

# Positive canonical femur angle = femur swings FORWARD.
FEMUR_SIGN = {
    'FL': +1,   # 13: + = forward
    'BL': +1,   # 14: + = forward
    'FR': -1,   # 15: - = forward
    'BR': -1,   # 16: - = forward
}

# Positive canonical knee angle = tibia swings FORWARD (relative to femur).
KNEE_SIGN = {
    'FR': -1,   # 17: - = forward
    'BR': -1,   # 18: - = forward
    'FL': +1,   # 21: + = forward
    'BL': +1,   # 22: + = forward
}


def leg_ik(x, y, z, femur_length=FEMUR_LENGTH, tibia_length=TIBIA_LENGTH,
           coxa_length=COXA_LENGTH):
    """
    Compute canonical (sign-independent) hip, femur, knee angles for a
    desired foot position (x, y, z) in the leg-local frame described above,
    measured from the hip-roll axis.

    Returns (hip_angle, femur_angle, knee_angle) in radians, using the
    CANONICAL sign convention (positive femur = forward, positive knee =
    forward, positive hip = outward). Multiply by the per-leg sign tables
    above to get the actual command to send to that leg's real joint.

    Raises ValueError if the target is unreachable.
    """
    # Hip abduction angle: rotate around x-axis to point at (y, z)
    hip_angle = math.atan2(y, -z)  # -z because z is "up", leg hangs down

    # Distance from hip to foot in the (rotated) leg plane, after removing
    # the coxa offset and the lateral (hip-abduction) component
    horiz_dist = math.sqrt(y**2 + z**2) - coxa_length
    leg_reach = math.sqrt(x**2 + horiz_dist**2)

    max_reach = femur_length + tibia_length
    if leg_reach > max_reach * 0.999:
        raise ValueError(f"Target unreachable: reach={leg_reach:.4f} > max={max_reach:.4f}")
    if leg_reach < abs(femur_length - tibia_length) * 1.001:
        raise ValueError(f"Target too close: reach={leg_reach:.4f}")

    # Law of cosines for knee angle
    cos_knee = (femur_length**2 + tibia_length**2 - leg_reach**2) / (2 * femur_length * tibia_length)
    cos_knee = max(-1.0, min(1.0, cos_knee))
    knee_interior = math.acos(cos_knee)
    
    # Original geometric solution (positive knee angle, subtract femur offset)
    knee_angle = math.pi - knee_interior

    # Femur angle: angle to target plus angle contributed by knee bend
    angle_to_target = math.atan2(x, horiz_dist)
    cos_femur_offset = (femur_length**2 + leg_reach**2 - tibia_length**2) / (2 * femur_length * leg_reach)
    cos_femur_offset = max(-1.0, min(1.0, cos_femur_offset))
    femur_offset = math.acos(cos_femur_offset)
    
    femur_angle = angle_to_target - femur_offset

    return hip_angle, femur_angle, knee_angle


def leg_ik_to_command(leg_name, x, y, z):
    """
    Full pipeline: canonical IK -> per-leg signed joint commands.
    Returns dict {joint_name: angle_command} for this leg's 3 joints.
    """
    hip_c, femur_c, knee_c = leg_ik(x, y, z)

    names = LEGS[leg_name]
    return {
        names['hip']:   HIP_SIGN[leg_name] * hip_c,
        names['femur']: FEMUR_SIGN[leg_name] * femur_c,
        names['knee']:  KNEE_SIGN[leg_name] * knee_c,
    }


# Neutral standing foot position (relative to hip), same for all 4 legs by
# symmetry. Tune STANCE_Z (leg length when standing) and STANCE_X (default
# fore/aft foot placement) to match a stable stance in your sim.
STANCE_X = 0.0
STANCE_Y = 0.0
STANCE_Z = -0.16   # ~85% of max reach (0.1097+0.1174=0.2271), slightly bent knee