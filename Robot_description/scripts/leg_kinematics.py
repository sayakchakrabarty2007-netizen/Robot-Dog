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
COXA_LENGTH = 0.027  # coxa (hip roll to hip pitch distance along Y in URDF is 0.027m)

# IK configuration
# The math yields 2 valid solutions (knee bends forward vs backward). 
# User requested << configuration (all knees point forward).
KNEE_BEND_DIR = {
    'FL': 1.0,
    'FR': 1.0,
    'BL': 1.0,
    'BR': 1.0
}

# ---- per-leg joint names (from your# URDF joint names for each leg
# The URDF Z axis represents Left(+)/Right(-).
# Front holder (X=+0.1025) has Revolute 48(FR, Z=-) and 49(FL, Z=+).
# Back holder (X=-0.1025) has Revolute 50(BL, Z=+) and 51(BR, Z=-).
LEGS = {
    'FL': {'hip': 'Revolute 49', 'femur': 'Revolute 14', 'knee': 'Revolute 22'},
    'FR': {'hip': 'Revolute 48', 'femur': 'Revolute 16', 'knee': 'Revolute 18'},
    'BL': {'hip': 'Revolute 50', 'femur': 'Revolute 13', 'knee': 'Revolute 21'},
    'BR': {'hip': 'Revolute 51', 'femur': 'Revolute 15', 'knee': 'Revolute 17'},
}

# ---- sign conventions from YOUR manual testing ----
# Positive canonical hip angle = leg abducts OUTWARD (away from body).
HIP_SIGN = {
    'FR': -1,   # 51: + = inside -> outside needs negative canonical->cmd
    'FL': +1,   # 50: + = outside matches canonical directly
    'BR': -1,   # 48: + = inside, - = outside  -> flip
    'BL': +1,   # 49: + = outside matches canonical directly
}

# Positive canonical femur angle = femur swings FORWARD.
FEMUR_SIGN = {
    'FL': -1,
    'FR': 1,
    'BL': -1,
    'BR': 1,
}

# Positive canonical knee angle = tibia swings FORWARD (relative to femur).
KNEE_SIGN = {
    'FL': 1,
    'FR': -1,
    'BL': 1,
    'BR': -1,
}


def leg_ik(x, y, z, femur_length=FEMUR_LENGTH, tibia_length=TIBIA_LENGTH,
           coxa_length=COXA_LENGTH, knee_bend_dir=1.0):
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
    # Exact analytical IK for a leg with a lateral offset (coxa_length)
    L_sq = y**2 + z**2
    if L_sq < coxa_length**2:
        raise ValueError(f"Target too close laterally: L={math.sqrt(L_sq):.4f} < coxa={coxa_length}")
    
    L_vert = math.sqrt(abs(L_sq - coxa_length**2))
    
    # Hip angle (roll) - Note: z is negative, so -z is positive (pointing down relative to hip)
    hip_angle = math.atan2(y, -z) - math.atan2(coxa_length, L_vert)
    
    # 2. In the X-(YZ) plane, calculate femur and knee angles
    leg_reach = math.sqrt(x**2 + L_vert**2)
    
    # Knee interior angle using Law of Cosines
    cos_knee = (femur_length**2 + tibia_length**2 - leg_reach**2) / (2 * femur_length * tibia_length)
    cos_knee = max(min(cos_knee, 1.0), -1.0) # Clamp to avoid domain errors
    knee_interior = math.acos(cos_knee)
    
    # Canonical knee angle: 0 means straight, positive means bent
    knee_angle = knee_bend_dir * (math.pi - knee_interior)
    
    # Femur angle using Law of Cosines
    angle_to_target = math.atan2(x, L_vert)
    cos_femur = (femur_length**2 + leg_reach**2 - tibia_length**2) / (2 * femur_length * leg_reach)
    cos_femur = max(min(cos_femur, 1.0), -1.0)
    femur_offset = math.acos(cos_femur)
    
    femur_angle = angle_to_target + knee_bend_dir * femur_offset
    
    return hip_angle, femur_angle, knee_angle


def leg_ik_to_command(leg_name, x, y, z):
    """
    Full pipeline: canonical IK -> per-leg signed joint commands.
    Returns dict {joint_name: angle_command} for this leg's 3 joints.
    """
    bend_dir = KNEE_BEND_DIR[leg_name]
    hip_c, femur_c, knee_c = leg_ik(x, y, z, knee_bend_dir=bend_dir)

    names = LEGS[leg_name]
    
    # No structural offsets needed because the real robot is assembled
    # with straight legs at the zero position.
    return {
        names['hip']:   HIP_SIGN[leg_name] * hip_c,
        names['femur']: FEMUR_SIGN[leg_name] * (femur_c - 0.522),
        names['knee']:  KNEE_SIGN[leg_name] * (knee_c - 1.042),
    }


# Neutral standing foot position (relative to hip), same for all 4 legs by
# symmetry. Tune STANCE_Z (leg length when standing) and STANCE_X (default
# fore/aft foot placement) to match a stable stance in your sim.
STANCE_X = 0.0
STANCE_Y = 0.0
STANCE_Z = -0.16   # ~85% of max reach (0.1097+0.1174=0.2271), slightly bent knee