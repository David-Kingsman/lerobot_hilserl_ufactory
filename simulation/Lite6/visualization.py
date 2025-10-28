import pybullet as p
import pybullet_data
import time
import math
import os

# 1) connect GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=50, cameraPitch=-30, cameraTargetPosition=[0,0,0.3])

# 2) load URDF, enable self-collision
urdf_path = "/home/zekaijin/lerobot-hilserl-ufactory/simulation/Lite6/lite6.urdf"

assert os.path.exists(urdf_path), f"URDF not found: {urdf_path}"
flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
robot = p.loadURDF(urdf_path, basePosition=[0,0,0], useFixedBase=True, flags=flags)

# 3) print joint information
num_joints = p.getNumJoints(robot)
print("num_joints:", num_joints)
for i in range(num_joints):
    info = p.getJointInfo(robot, i)
    print(i, info[1].decode(), "type:", info[2], "parent:", info[16], "child link:", info[12].decode())

# 4) set joint angle to your "neutral/initial pose"
# example: joint1..joint6
neutral = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# neutral = [0.0401, 0.5184, 1.0629, 3.1940, -0.6126, -3.8101]
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
            # 规范化顺序，避免重复
            pairs.add(tuple(sorted((a, b))))
    if pairs:
        print("Self-collisions:", sorted(pairs))
    time.sleep(0.02)