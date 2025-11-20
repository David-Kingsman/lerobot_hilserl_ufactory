# HIL-SERL Real Robot Training Workflow Guide

In this tutorial you will go through the full Human-in-the-Loop Sample-Efficient Reinforcement Learning (HIL-SERL) workflow using LeRobot. You will master training a policy with RL on a real robot in just a few hours.

HIL-SERL is a sample-efficient reinforcement learning algorithm that combines human demonstrations with online learning and human interventions. The approach starts from a small set of human demonstrations, uses them to train a reward classifier, and then employs an actor-learner architecture where humans can intervene during policy execution to guide exploration and correct unsafe behaviors. In this tutorial, you'll use a gamepad to provide interventions and control the robot during the learning process.

It combines three key ingredients:

1. **Offline demonstrations & reward classifier:** a handful of human-teleop episodes plus a vision-based success detector give the policy a shaped starting point.

2. **On-robot actor / learner loop with human interventions:** a distributed Soft Actor Critic (SAC) learner updates the policy while an actor explores on the physical robot; the human can jump in at any time to correct dangerous or unproductive behaviour.

3. **Safety & efficiency tools:** joint/end-effector (EE) bounds, crop region of interest (ROI) preprocessing and WandB monitoring keep the data useful and the hardware safe.

Together these elements let HIL-SERL reach near-perfect task success and faster cycle times than imitation-only baselines.

<p align="center">
  <img
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/hilserl-main-figure.png"
    alt="HIL-SERL workflow"
    title="HIL-SERL workflow"
    width="100%"
  ></img>
</p>

<p align="center">
  <i>HIL-SERL workflow, Luo et al. 2024</i>
</p>

This guide provides step-by-step instructions for training a robot policy using LeRobot's HilSerl implementation to train on a real robot.

## What do I need?

- A gamepad (recommended) or keyboard to control the robot
- A Nvidia GPU
- A real robot with any teleoperation methods
- A URDF file for the robot for the kinematics package (check `lerobot/model/kinematics.py`)

## Install LeRobot with HIL-SERL

To install LeRobot with HIL-SERL, you need to install the `hilserl` extra.

```bash
pip install -e ".[hilserl]"
```

## Velocity control and position control teleoperation supporting Ufactory robot Xarm6, Lite6 and Ufactory 850

```bash
# Example for position control on gamepad using Xarm6
PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.scripts.lerobot_teleoperate \
  --robot.type=xarm6_end_effector \
  --robot.ip=192.168.1.235 \
  --robot.act_features=xyz_delta \
  --robot.use_gripper=true \
  --teleop.type=gamepad \
  --teleop.use_gripper=true 

# Example for velocity control on gamepad using Xarm6
PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.scripts.lerobot_teleoperate \
  --robot.type=xarm6_end_effector_hil \
  --robot.ip=192.168.1.235 \
  --robot.act_features=xyz_delta \
  --robot.use_gripper=true \
  --teleop.type=gamepad \
  --teleop.use_gripper=true

** robot type: xarm_end_effector, xarm6_end_effector, xarm6_end_effector_hil, xarm ... **
** teleop options: spacemouse (6dof), keyboard_ee (6dof), gamepad (3dof) **
** add " --display_data=true "   and then you can see the robot state in the console and Rerun **
```

## Check cameras

This script helps you determine the port of the existing cameras.

```bash
cd /home/zekaijin/lerobot-hilserl-ufactory && PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.scripts.lerobot_find_cameras opencv
```

#### Configuration Examples

**Basic Observation Processing**:

```json
{
  "env": {
    "processor": {
      "observation": {
        "add_joint_velocity_to_observation": true,
        "add_current_to_observation": false,
        "display_cameras": false
      }
    }
  }
}
```

**Image Processing**:

```json
{
  "env": {
    "processor": {
      "image_preprocessing": {
        "crop_params_dict": {
          "observation.images.front": [180, 250, 120, 150],
          "observation.images.side": [180, 207, 180, 200]
        },
        "resize_size": [128, 128]
      }
    }
  }
}
```

**Inverse Kinematics Setup**:

```json
{
  "env": {
    "processor": {
      "inverse_kinematics": {
        "urdf_path": "path/to/robot.urdf",
        "target_frame_name": "end_effector",
        "end_effector_bounds": {
          "min": [0.16, -0.08, 0.03],
          "max": [0.24, 0.2, 0.1]
        },
        "end_effector_step_sizes": {
          "x": 0.02,
          "y": 0.02,
          "z": 0.02
        }
      }
    }
  }
}
```

### Advanced Observation Processing

The HIL-SERL framework   additional observation processing features that can improve policy learning:

#### Joint Velocity Processing

Enable joint velocity estimation to provide the policy with motion information:

```json
{
  "env": {
    "processor": {
      "observation": {
        "add_joint_velocity_to_observation": true
      }
    }
  }
}
```

This processor:

- Estimates joint velocities using finite differences between consecutive joint position readings
- Adds velocity information to the observation state vector
- Useful for policies that need motion awareness for dynamic tasks

#### Motor Current Processing

Monitor motor currents to detect contact forces and load conditions:

```json
{
  "env": {
    "processor": {
      "observation": {
        "add_current_to_observation": true
      }
    }
  }
}
```

This processor:

- Reads motor current values from the robot's control system
- Adds current measurements to the observation state vector
- Helps detect contact events, object weights, and mechanical resistance
- Useful for contact-rich manipulation tasks

#### Combined Observation Processing

You can enable multiple observation processing features simultaneously:

```json
{
  "env": {
    "processor": {
      "observation": {
        "add_joint_velocity_to_observation": true,
        "add_current_to_observation": true,
        "add_ee_pose_to_observation": false,
        "display_cameras": false
      }
    }
  }
}
```

**Note**: Enabling additional observation features increases the state space dimensionality, which may require adjusting your policy network architecture and potentially collecting more training data.


### Finding Robot Workspace Bounds

Before collecting demonstrations, you need to determine the appropriate operational bounds for your robot.

This helps simplify the problem of learning on the real robot in two ways: 1) by limiting the robot's operational space to a specific region that solves the task and avoids unnecessary or unsafe exploration, and 2) by allowing training in end-effector space rather than joint space. Empirically, learning in joint space for reinforcement learning in manipulation is often a harder problem - some tasks are nearly impossible to learn in joint space but become learnable when the action space is transformed to end-effector coordinates.


## Using lerobot-find-joint-limits**

This script helps you find the safe operational bounds for your robot's end-effector. Given that you have a follower and leader arm, you can use the script to find the bounds for the follower arm that will be applied during training.
Bounding the action space will reduce the redundant exploration of the agent and guarantees safety.

```bash
# gamepad test
PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.scripts.lerobot_find_joint_limits_gamepad  \
  --robot.type=xarm6_end_effector_hil  \
  --robot.ip=192.168.1.235  \
  --teleop.type=gamepad  \
  --teleop_time_s=30

# spacemouse test
PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.scripts.lerobot_find_joint_limits_spacemouse \
  --robot.type=lite6_end_effector_hil \
  --robot.ip=192.168.1.193 \
  --teleop.type=spacemouse \
  --teleop_time_s=30

# meta quest test
cd experiments/arm_control_base
python scripts/find_joint_limits_metaquest.py \
  --robot_ip=192.168.1.193 \
  --teleop_time_s=30
```

**Workflow**

This script will:
1. Connect to UFactory Lite6 robot
2. Connect to Meta Quest 2 controller, gamepad, spacemouse
3. Find the joint and end-effector position limits through Meta Quest 2 control or gamepad and spacemouse control
4. Output the found limits, format same as LeRobot

   ```
    UFactory Lite6 + Meta Quest 2 joint limit
    ============================================================
    Max ee position: [388.2588, 86.6573, 213.5001]
    Min ee position: [185.3618, -91.9344, 110.2758]
    Max joint pos position: [13.8942, 63.5855, 101.24, 30.9897, 42.0742, -20.4426, 0.0]
    Min joint pos position: [-20.4989, 22.5459, 32.6588, -23.8765, 8.5189, -64.0866, 0.0]

    Generated time: 2025-09-25 15:53:07
    Control method: Meta Quest 2
    Robot arm: UFactory Lite6
    
   ```
3. Use these values in the configuration of your teleoperation device (TeleoperatorConfig) under the `end_effector_bounds` field

**Example Configuration**

```json
"end_effector_bounds": {
    "max": [388.2588, 86.6573, 213.5001],
    "min": [185.3618, -91.9344, 110.2758]
}
```

## Collecting Demonstrations

With the bounds defined, you can safely collect demonstrations for training. Training RL with off-policy algorithm allows us to use offline datasets collected in order to improve the efficiency of the learning process.

**Setting Up Record Mode**

1. Set `mode` to `"record"` at the root level
2. Specify a unique `repo_id` for your dataset in the `dataset` section (e.g., "username/task_name")
3. Set `num_episodes_to_record` in the `dataset` section to the number of demonstrations you want to collect
4. Set `env.processor.image_preprocessing.crop_params_dict` to `{}` initially (we'll determine crops later)
5. Configure `env.robot`, `env.teleop`, and other hardware settings in the `env` section


**Recording Demonstrations**

Start the recording process, 

```bash
# Ufactory lite6 using spacemouse (6dof)
PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.rl.gym_manipulator \
  --config configs/ufactory/lite6/env_config_hilserl_lite6_spacemouse.json

# Ufactory lite6 using gamepad (3dof)
PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.rl.gym_manipulator  \
--config configs/ufactory/lite6/env_config_hilserl_lite6_gamepad.json

# Ufactory xarm6 using gamepad (3dof)
PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.rl.gym_manipulator 
  --config configs/ufactory/xarm6/env_config_hilserl_xarm6_gamepad.json \
```

During recording:

1. The robot will reset to the initial position defined in the configuration file `env.processor.reset.fixed_reset_joint_positions`
2. Complete the task successfully
3. The episode ends with a reward of 1 when you press the "success" button
4. If the time limit is reached, or the fail button is pressed, the episode ends with a reward of 0
5. You can rerecord an episode by pressing the "rerecord" button
6. The process automatically continues to the next episode
7. After recording all episodes, the dataset is pushed to the Hugging Face Hub (optional) and saved locally

### Processing the Dataset

After collecting demonstrations, process them to determine optimal camera crops.
Reinforcement learning is sensitive to background distractions, so it is important to crop the images to the relevant workspace area.

Visual RL algorithms learn directly from pixel inputs, making them vulnerable to irrelevant visual information. Background elements like changing lighting, shadows, people moving, or objects outside the workspace can confuse the learning process. Good ROI selection should:

- Include only the essential workspace where the task happens
- Capture the robot's end-effector and all objects involved in the task
- Exclude unnecessary background elements and distractions

Note: If you already know the crop parameters, you can skip this step and just set the `crop_params_dict` in the configuration file during recording.

## Determining Crop Parameters

Use the `crop_dataset_roi.py` script to interactively select regions of interest in your camera images:

```bash
PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.rl.crop_dataset_roi \
  --root  datasets/xarm6_pick_cube_test2
```

1. For each camera view, the script will display the first frame
2. Draw a rectangle around the relevant workspace area
3. Press 'c' to confirm the selection
4. Repeat for all camera views
5. The script outputs cropping parameters and creates a new cropped dataset

Example output:

```
Selected Rectangular Regions of Interest (top, left, height, width):
observation.images.webcam_1: (13, 47, 64, 66)
observation.images.realsense: (9, 10, 81, 93)

Converted ROI coordinates for original image size (640x480):
Format: (top, left, height, width)
observation.images.webcam_1: (48, 235, 240, 330)
observation.images.realsense: (33, 50, 303, 465)
```

**Recommended image resolution**

Most vision-based policies have been validated on square inputs of either **128×128** (default) or **64×64** pixels. We therefore advise setting the resize_size parameter to [128, 128] – or [64, 64] if you need to save GPU memory and bandwidth. Other resolutions are possible but have not been extensively tested.

### Training a Reward Classifier

The reward classifier plays an important role in the HIL-SERL workflow by automating reward assignment and automatically detecting episode success. Instead of manually defining reward functions or relying on human feedback for every timestep, the reward classifier learns to predict success/failure from visual observations. This enables the RL algorithm to learn efficiently by providing consistent and automated reward signals based on the robot's camera inputs.

This guide explains how to train a reward classifier for human-in-the-loop reinforcement learning implementation of LeRobot. Reward classifiers learn to predict the reward value given a state which can be used in an RL setup to train a policy.

**Note**: Training a reward classifier is optional. You can start the first round of RL experiments by annotating the success manually with your gamepad or keyboard device.

The reward classifier implementation in `modeling_classifier.py` uses a pretrained vision model to process the images. It can output either a single value for binary rewards to predict success/fail cases or multiple values for multi-class settings.

## Collecting a Dataset for the reward classifier

Before training, you need to collect a dataset with labeled examples. The `record_dataset` function in `gym_manipulator.py` enables the process of collecting a dataset of observations, actions, and rewards.

To collect a dataset, you need to modify some parameters in the environment configuration based on HILSerlRobotEnvConfig.

```bash
PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.rl.gym_manipulator \
  --config_path configs/ufactory/xarm6/reward_classifier_train_config_xarm6.json 
```

**Key Parameters for Data Collection**

The `env.processor.reset.terminate_on_success` parameter allows you to control episode termination behavior. When set to `false`, episodes will continue even after success is detected, allowing you to collect more positive examples with the reward=1 label. This is crucial for training reward classifiers as it provides more success state examples in your dataset. When set to `true` (default), episodes terminate immediately upon success detection.

**Important**: For reward classifier training, set `terminate_on_success: false` to collect sufficient positive examples. For regular HIL-SERL training, keep it as `true` to enable automatic episode termination when the task is completed successfully.


## Training the Classifier

To train the classifier, use the `train.py` script with your configuration and add the reward classifier pretrain:

```bash
lerobot-train --config_path path/to/reward_classifier_train_config.json
```

**Example workflow**:

1. **Collect a dataset**:

   ```bash
  PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.rl.gym_manipulator 
    --config configs/ufactory/xarm6/env_config_hilserl_xarm6_gamepad.json \
   ```

2. **Train the classifier**:

   ```bash
   PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.scripts.lerobot_train \
    --config_path configs/ufactory/xarm6/reward_classifier_train_config_xarm6.json
   ```

3. **Use in HiLSERL**: Add the trained classifier path to `env.processor.reward_classifier.pretrained_path` in your HiLSERL config.

### Training with Actor-Learner

The LeRobot system uses a distributed actor-learner architecture for training. This architecture decouples robot interactions from the learning process, allowing them to run concurrently without blocking each other. The actor server handles robot observations and actions, sending interaction data to the learner server. The learner server performs gradient descent and periodically updates the actor's policy weights. You will need to start two processes: a learner and an actor.

**Configuration Setup**

 The training config is based on the main `TrainRLServerPipelineConfig` class in `lerobot/configs/train.py`.

1. Configure the policy settings (`type="sac"`, `device`, etc.)
2. Set `dataset` to your cropped dataset
3. Configure environment settings with crop parameters
4. Check the other parameters related to SAC 
5. Verify that the `policy` config is correct with the right `input_features` and `output_features` for your task.

**Starting the Learner**

First, start the learner server process:

```bash
PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.rl.learner --config_path configs/ufactory/xarm6/train_config_hilserl_xarm6.json
```

The learner:

- Initializes the policy network
- Prepares replay buffers
- Opens a `gRPC` server to communicate with actors
- Processes transitions and updates the policy

**Starting the Actor**

In a separate terminal, start the actor process with the same configuration:

```bash
PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.rl.actor --config_path configs/ufactory/xarm6/train_config_hilserl_xarm6.json
```

The actor:

- Connects to the learner via `gRPC`
- Initializes the environment
- Execute rollouts of the policy to collect experience
- Sends transitions to the learner
- Receives updated policy parameters

**Training Flow**

The training proceeds automatically:

1. The actor executes the policy in the environment
2. Transitions are collected and sent to the learner
3. The learner updates the policy based on these transitions
4. Updated policy parameters are sent back to the actor
5. The process continues until the specified step limit is reached

**Human in the Loop**

- The key to learning efficiently is to have human interventions to provide corrective feedback and completing the task to aide the policy learning and exploration.
- To perform human interventions, you can press the upper right trigger button on the gamepad (or the `space` key on the keyboard). This will pause the policy actions and allow you to take over.
- A successful experiment is one where the human has to intervene at the start but then reduces the amount of interventions as the policy improves. You can monitor the intervention rate in the `wandb` dashboard.

<p align="center">
  <img
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/hil_effect.png?raw=true"
    alt="Figure shows the control mappings on a Logitech gamepad."
    title="Gamepad Control Mapping"
    width="100%"
  ></img>
</p>

<p align="center">
  <i>
    Example showing how human interventions help guide policy learning over time
  </i>
</p>

- The figure shows the plot of the episodic reward over interaction step. The figure shows the effect of human interventions on the policy learning.
- The orange curve is an experiment without any human interventions. While the pink and blue curves are experiments with human interventions.
- We can observe that the number of steps where the policy starts achieving the maximum reward is cut by a quarter when human interventions are present.

**Monitoring and Debugging**

If you have `wandb.enable` set to `true` in your configuration, you can monitor training progress in real-time through the [Weights & Biases](https://wandb.ai/site/) dashboard.

### Guide to Human Interventions

The learning process is very sensitive to the intervention strategy. It will takes a few runs to understand how to intervene effectively. Some tips and hints:

- Allow the policy to explore for a few episodes at the start of training.
- Avoid intervening for long periods of time. Try to intervene in situation to correct the robot's behaviour when it goes off track.
- Once the policy starts achieving the task, even if its not perfect, you can limit your interventions to simple quick actions like a simple grasping commands.

The ideal behaviour is that your intervention rate should drop gradually during training as shown in the figure below.

<p align="center">
  <img
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/intervention_rate_tutorial_rl.png?raw=true"
    alt="Intervention rate"
    title="Intervention rate during training"
    width="100%"
  ></img>
</p>

<p align="center">
  <i>
    Plot of the intervention rate during a training run on a pick and lift cube
    task
  </i>
</p>

### Key hyperparameters to tune

Some configuration values have a disproportionate impact on training stability and speed:

- **`temperature_init`** (`policy.temperature_init`) – initial entropy temperature in SAC. Higher values encourage more exploration; lower values make the policy more deterministic early on. A good starting point is `1e-2`. We observed that setting it too high can make human interventions ineffective and slow down learning.
- **`policy_parameters_push_frequency`** (`policy.actor_learner_config.policy_parameters_push_frequency`) – interval in _seconds_ between two weight pushes from the learner to the actor. The default is `4 s`. Decrease to **1-2 s** to provide fresher weights (at the cost of more network traffic); increase only if your connection is slow, as this will reduce sample efficiency.
- **`storage_device`** (`policy.storage_device`) – device on which the learner keeps the policy parameters. If you have spare GPU memory, set this to `"cuda"` (instead of the default `"cpu"`). Keeping the weights on-GPU removes CPU→GPU transfer overhead and can significantly increase the number of learner updates per second.




Paper citation:

```
@article{luo2024precise,
  title={Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning},
  author={Luo, Jianlan and Xu, Charles and Wu, Jeffrey and Levine, Sergey},
  journal={arXiv preprint arXiv:2410.21845},
  year={2024}
}
```
