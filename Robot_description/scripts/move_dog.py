#!/usr/bin/env python3
"""
Interactive Joint Tester for Quadruped Robot (Phase 1)
=====================================================
Lets you select ONE joint at a time and move it to verify which joint is which.
All other joints hold at position 0 (standing pose).

Usage:
  python3 move_dog.py

Controls:
  - Type a joint NUMBER (1-16) to select it
  - Type '+' to move the selected joint by +0.1 rad
  - Type '-' to move the selected joint by -0.1 rad
  - Type '0' to reset the selected joint to 0
  - Type 'r' to reset ALL joints to 0
  - Type 'q' to quit
"""
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import sys
import threading

# All 16 controllable joints in order
JOINT_NAMES = [
    'Revolute 48',   # 1  - Hip (shoulder roll)
    'Revolute 49',   # 2  - Hip (shoulder roll)
    'Revolute 50',   # 3  - Hip (shoulder roll)
    'Revolute 51',   # 4  - Hip (shoulder roll)
    'Revolute 13',   # 5  - Femur (hip pitch)
    'Revolute 14',   # 6  - Femur (hip pitch)
    'Revolute 15',   # 7  - Femur (hip pitch)
    'Revolute 16',   # 8  - Femur (hip pitch)
    'Revolute 17',   # 9  - Tibia/Knee
    'Revolute 18',   # 10 - Tibia/Knee
    'Revolute 21',   # 11 - Tibia/Knee
    'Revolute 22',   # 12 - Tibia/Knee
    'Revolute 24',   # 13 - Knee linkage plate
    'Revolute 25',   # 14 - Knee linkage plate
    'Revolute 26',   # 15 - Knee linkage plate
    'Revolute 27',   # 16 - Knee linkage plate
]


class JointTester(Node):
    def __init__(self):
        super().__init__('joint_tester')
        self.publisher_ = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        # All joints start at 0 (standing)
        self.positions = [0.0] * len(JOINT_NAMES)
        self.selected_joint = 0  # index into JOINT_NAMES
        self.step = 0.1  # radians per keypress

    def send_trajectory(self):
        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)

        point = JointTrajectoryPoint()
        point.positions = list(self.positions)
        point.time_from_start = Duration(sec=0, nanosec=500000000)  # 0.5s
        msg.points.append(point)

        self.publisher_.publish(msg)

    def print_status(self):
        print("\n" + "=" * 60)
        print("  JOINT TESTER - Phase 1: Identify Joints")
        print("=" * 60)
        for i, name in enumerate(JOINT_NAMES):
            marker = " >>>" if i == self.selected_joint else "    "
            pos = self.positions[i]
            bar_len = int(abs(pos) / 0.1)
            bar = ("+" * bar_len) if pos >= 0 else ("-" * bar_len)
            print(f"  {marker} [{i+1:2d}] {name:20s} = {pos:+.2f} rad  {bar}")
        print("-" * 60)
        print(f"  Selected: [{self.selected_joint+1}] {JOINT_NAMES[self.selected_joint]}")
        print(f"  Step size: {self.step:.2f} rad")
        print("-" * 60)
        print("  Commands:")
        print("    1-16  = Select joint")
        print("    +/-   = Move selected joint")
        print("    0     = Zero selected joint")
        print("    r     = Reset ALL to 0")
        print("    s     = Change step size")
        print("    q     = Quit")
        print("=" * 60)
        print("> ", end="", flush=True)


def input_loop(tester):
    """Run in a separate thread to handle user input."""
    tester.print_status()
    # Send initial standing pose
    tester.send_trajectory()

    while rclpy.ok():
        try:
            cmd = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == 'q':
            print("Quitting...")
            rclpy.shutdown()
            break
        elif cmd == '+':
            tester.positions[tester.selected_joint] += tester.step
            tester.send_trajectory()
            tester.get_logger().info(
                f"Joint [{tester.selected_joint+1}] {JOINT_NAMES[tester.selected_joint]} "
                f"-> {tester.positions[tester.selected_joint]:+.2f} rad"
            )
        elif cmd == '-':
            tester.positions[tester.selected_joint] -= tester.step
            tester.send_trajectory()
            tester.get_logger().info(
                f"Joint [{tester.selected_joint+1}] {JOINT_NAMES[tester.selected_joint]} "
                f"-> {tester.positions[tester.selected_joint]:+.2f} rad"
            )
        elif cmd == '0':
            tester.positions[tester.selected_joint] = 0.0
            tester.send_trajectory()
            tester.get_logger().info(
                f"Joint [{tester.selected_joint+1}] {JOINT_NAMES[tester.selected_joint]} "
                f"-> zeroed"
            )
        elif cmd == 'r':
            tester.positions = [0.0] * len(JOINT_NAMES)
            tester.send_trajectory()
            tester.get_logger().info("ALL joints reset to 0")
        elif cmd == 's':
            print("Enter new step size (e.g. 0.05, 0.1, 0.2): ", end="", flush=True)
            try:
                val = float(input().strip())
                tester.step = val
            except ValueError:
                print("Invalid number")
        else:
            # Try to parse as a joint number
            try:
                num = int(cmd)
                if 1 <= num <= len(JOINT_NAMES):
                    tester.selected_joint = num - 1
                    tester.get_logger().info(
                        f"Selected joint [{num}] {JOINT_NAMES[num-1]}"
                    )
                else:
                    print(f"Joint number must be 1-{len(JOINT_NAMES)}")
            except ValueError:
                print(f"Unknown command: '{cmd}'")

        tester.print_status()


def main(args=None):
    rclpy.init(args=args)
    tester = JointTester()

    # Run input handling in a separate thread
    input_thread = threading.Thread(target=input_loop, args=(tester,), daemon=True)
    input_thread.start()

    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
