"""
This script is used to visualize the robots in PyBullet.

Usage:
    python simulation/visualization.py --robot lite6
    python simulation/visualization.py --robot xarm6
    python simulation/visualization.py --robot xarm7
    python simulation/visualization.py --robot uf850

Options:
    --robot: Robot model to visualize. Choices: lite6, xarm6, uf850. Default: lite6.
"""
import pybullet as p
import pybullet_data
import time
import math
import os
import argparse

parser = argparse.ArgumentParser(description="Visualize UFactory robots in PyBullet")
parser.add_argument("--robot", choices=["lite6", "xarm6", "xarm7", "uf850"], default="lite6", help="Robot model to visualize")
args = parser.parse_args()

# 1) connect GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=50, cameraPitch=-30, cameraTargetPosition=[0,0,0.3])

# 2) load URDF, enable self-collision
ROOT = "/home/zekaijin/lerobot-hilserl-ufactory"
URDF_MAP = {
    "lite6": f"{ROOT}/simulation/Lite6/lite6.urdf",
    "xarm6": f"{ROOT}/simulation/xarm6/xarm6.urdf",
    "xarm7": f"{ROOT}/simulation/xarm7/xarm7.urdf",
    "uf850": f"{ROOT}/simulation/uf850/uf850.urdf",
}
urdf_path = URDF_MAP[args.robot]

if not os.path.exists(urdf_path):
    raise FileNotFoundError(f"URDF not found for {args.robot}: {urdf_path}\nplease check the path or modify the URDF_MAP in visualization.py")

flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
robot = p.loadURDF(urdf_path, basePosition=[0,0,0], useFixedBase=True, flags=flags)

# 3) print joint information
num_joints = p.getNumJoints(robot)
print("num_joints:", num_joints)
for i in range(num_joints):
    info = p.getJointInfo(robot, i)
    print(i, info[1].decode(), "type:", info[2], "parent:", info[16], "child link:", info[12].decode())

# 4) set joint angle to your "neutral/initial pose"
# if the neutral pose is different for different robots, you can modify the NEUTRAL_MAP in visualization.py
NEUTRAL_MAP = {
    "lite6": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "xarm6": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "xarm7": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "uf850": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}
default_len = 7 if args.robot == "xarm7" else 6
neutral = NEUTRAL_MAP.get(args.robot, [0.0]*default_len)
for jid, q in enumerate(neutral):
    p.resetJointState(robot, jid, q)

p.setGravity(0, 0, -9.8)

# 5) loop, detect and print self-collisions
def format_link(name_index):
    link_idx = name_index
    if link_idx == -1:
        return "base_link"
    return p.getJointInfo(robot, link_idx)[12].decode()

print("Running... Close the window to stop.")
while True:
    p.stepSimulation()
    contacts = p.getContactPoints(bodyA=robot, bodyB=robot)
    pairs = set()
    for c in contacts:
        # c[3]: linkIndexA, c[4]: linkIndexB
        if c[3] != c[4]:
            a = format_link(c[3])
            b = format_link(c[4])
            # normalize the order to avoid duplicates
            pairs.add(tuple(sorted((a, b))))
    if pairs:
        print("Self-collisions:", sorted(pairs))
    time.sleep(0.02)