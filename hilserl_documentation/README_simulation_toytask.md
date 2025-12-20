# Train RL in Simulation

This guide explains how to use the `gym_hil` simulation environments as an alternative to real robots when working with the LeRobot framework for Human-In-the-Loop (HIL) reinforcement learning. `gym_hil` is a package that provides Gymnasium-compatible simulation environments specifically designed for Human-In-the-Loop reinforcement learning. These environments allow you to:

- Train policies in simulation to test the RL stack before training on real robots
- Collect demonstrations in sim using external devices like gamepads or keyboards
- Perform human interventions during policy learning

Currently, the main environment is a Franka Panda robot simulation based on MuJoCo, with tasks like picking up a cube.

## Intro

HIL-SERL is a framework for training state-of-the-art robotic manipulation policies via reinforcement learning. The pipeline proceeds in three stages. First, we teleoperate the robot to curate positive and negative examples and fit a binary reward classifier. Second, we gather a small set of demonstrations, which is seeded into the demonstration buffer at the onset of RL training. Third, during online training, the learned classifier supplies a sparse reward while a human operator supplies corrective interventions. Early in training, interventions are frequent to illustrate successful strategies from diverse states and to suppress undesirable behaviors; as policy performance improves—measured by success rate and cycle time—the intervention rate is progressively reduced. This schedule yields a data-efficient training process that transitions from human-guided exploration to largely autonomous policy optimization.

## Installation

First, install the `gym_hil` package within the LeRobot environment: 

```bash
pip install -e ".[hilserl]"
```

## Gym-hil Overview

A collection of gymnasium environments for Human-In-the-Loop (HIL) reinforcement learning, compatible with Hugging Face's LeRobot codebase.
The `gym-hil` package provides environments designed for human-in-the-loop reinforcement learning. The list of environments are integrated with external devices like gamepads and keyboards, making it easy to collect demonstrations and perform interventions during learning.

Currently available environments:
- **Franka Panda Robot**: A robotic manipulation environment for Franka Panda robot based on MuJoCo

**What is Human-In-the-Loop (HIL) RL?**

Human-in-the-Loop (HIL) Reinforcement Learning keeps a human inside the control loop while the agent is training. During every rollout, the policy proposes an action, but the human may instantly override it for as many consecutive steps as needed; the robot then executes the human's command instead of the policy's choice. This approach improves sample efficiency and promotes safer exploration, as corrective actions pull the system out of unrecoverable or dangerous states and guide it toward high-value behaviors.

<div align="center">
  <img src="../media/hil-rl-schema.png" alt="Human-in-the-Loop RL Schema" width="70%"/>
</div>


## Configuration

To use `gym_hil` with LeRobot, you need to create a configuration file. An example is provided [here](https://huggingface.co/datasets/lerobot/config_examples/resolve/main/rl/gym_hil/env_config.json). Key configuration sections include:

### Environment Type and Task
```json
{
  "env": {
    "type": "gym_manipulator",
    "name": "gym_hil",
    "task": "PandaPickCubeGamepad-v0",
    "fps": 10
  },
  "device": "cuda"
}
```
Available tasks:
- `PandaPickCubeBase-v0`: Basic environment
- `PandaPickCubeGamepad-v0`: With gamepad control
- `PandaPickCubeKeyboard-v0`: With keyboard control

### Processor Configuration

```json
{
  "env": {
    "processor": {
      "control_mode": "gamepad",
      "gripper": {
        "use_gripper": true,
        "gripper_penalty": -0.02
      },
      "reset": {
        "control_time_s": 15.0,
        "fixed_reset_joint_positions": [0.0, 0.195, 0.0, -2.43, 0.0, 2.62, 0.785]},
      "inverse_kinematics": {
        "end_effector_step_sizes": {"x": 0.025,"y": 0.025,"z": 0.025}
      }
    }
  }
}
```
Important parameters:
- `gripper.gripper_penalty`: Penalty for excessive gripper movement
- `gripper.use_gripper`: Whether to enable gripper control
- `inverse_kinematics.end_effector_step_sizes`: Size of the steps in the x,y,z axes of the end-effector
- `control_mode`: Set to `"gamepad"` to use a gamepad controller

## Running with HIL RL of LeRobot

### Basic Usage

To run the environment, set mode to null:

```bash
# franka panda gamepad control (3dof)
PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src \
python -m lerobot.rl.gym_manipulator --config_path configs/simulation/gym_hil_env.json
# meta quest control (6dof)
PYTHONPATH=lerobot/src python -m lerobot.rl.gym_manipulator     --config_path configs/simulation/acfql/pick_and_lift/gym_hil_env_metaquest.json
```

### 3.2 Recording a Dataset

To collect a dataset, set the mode to `record` whilst defining the repo_id and number of episodes to record:

```json
{
  "env": {
    "type": "gym_manipulator",
    "name": "gym_hil",
    "task": "PandaPickCubeGamepad-v0"
  },
  "dataset": {
    "repo_id": "username/sim_dataset",
    "root": null,                  
    "task": "pick_cube",
    "num_episodes_to_record": 10,
    "replay_episode": null,
    "push_to_hub": true
  },
  "mode": "record"
}
```

### 3.3 Training a Policy

To train a policy, checkout the example json in `train_gym_hil_env.json` and run the actor and learner servers:
```shell
# actor servers
PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.rl.actor --config_path configs/simulation/train_gym_hil_env.json
```
Next, open a different terminal, run the learner server:
```shell 
# learner servers
PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.rl.learner --config_path configs/simulation/train_gym_hil_env.json
```

The simulation environment provides a safe and repeatable way to develop zand test your Human-In-the-Loop reinforcement learning components before deploying to real robots. 

### 3.4 Demo Video

<div align="center">
  <a href="https://www.youtube.com/watch?v=99sVWGECBas">
    <img src="https://img.youtube.com/vi/99sVWGECBas/maxresdefault.jpg" alt="Watch the gym-hil demo video" width="480"/>
  </a>
  <br/>
  <em>Click the image to watch a demo of gym-hil in action!</em>
</div>

#### Local Simulation Video 

If you are viewing the document in a local environment, you can directly play or download the local sample video:

<div align="center">
  <video width="640" controls>
    <source src="../media/hilserl_simulation.mp4" type="video/mp4" />
    Your browser does not support the video tag. You can download it instead:
    <a href="../media/hilserl_simulation.mp4">Download hilserl_simulation.mp4</a>
  </video>
  <br/>
  <a href="../media/hilserl_simulation.mp4">Download hilserl_simulation.mp4</a>
  
</div>

We use [HIL-SERL](https://hil-serl.github.io/) from [LeRobot](https://github.com/huggingface/lerobot) to train this policy.
The policy was trained for **10 minutes** with human in the loop.
After only 10 minutes of training, the policy successfully performs the task.

## 4. Effect of the Reward Classifier on Training Stability (Simulation Evidence)

Under the same `gym_hil` simulation and the same SAC setup, we compared two runs in Weights & Biases (wandb): one without the reward classifier (without RC) and one with the reward classifier (with RC). Key observations:

- Temperature and its gradient (`train/temperature`, `train/temperature_grad_norm`) are lower and smoother with RC, indicating a faster transition to a more deterministic policy.
- Loss and gradients (`train/loss_critic`, `train/loss_discrete_critic`, and actor/critic grad norms) are lower and more stable with RC, showing steadier updates.
- Sampling efficiency improves with RC, reflected by larger `train/replay_buffer_size` and higher `Policy frequency [Hz]`.
- Both runs reach `Episodic reward = 1`, but the with-RC run achieves stable success at fewer interaction steps.

Screenshots of the two experiments:

- Without reward classifier (baseline)
  ![without reward classifier](../media/wanb_without_reward_classifier.png)
- With reward classifier
  ![with reward classifier](../media/wanb_with_reward_classifier.png)

Practical tips:

- If curves stabilize early (e.g., around 80k steps) with `Episodic reward = 1`, consider using that checkpoint and stopping training earlier.
- For the classifier itself, increase the confidence threshold or reduce `number_of_steps_after_success` to avoid premature determinism and reward stretching biases.

# Training a Reward Classifier with LeRobot

This guide explains how to train a reward classifier for human-in-the-loop reinforcement learning implementation of  LeRobot. Reward classifiers learn to predict the reward value given a state which can be used in an RL setup to train a policy.


The reward classifier implementation in `modeling_classifier.py` uses a pretrained vision model to process the images. It can output either a single value for binary rewards to predict success/fail cases or multiple values for multi-class settings.

## 1. Collecting a Dataset
Before training, you need to collect a dataset with labeled examples. The `record_dataset` function in `gym_manipulator.py` enables the process of collecting a dataset of observations, actions, and rewards.

To collect a dataset, you need to modeify some parameters in the environment configuration based on HILSerlRobotEnvConfig.

```shell
   PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.scripts.lerobot_train \
    --config_path configs/simulation/reward_classifier_train_config.json
```

### Key Parameters for Data Collection:

- **mode**: set it to "record" to collect a dataset
- **repo_id**: "hf_username/dataset_name", name of the dataset and repo on the hub
- **num_episodes**: Number of episodes to record
- **number_of_steps_after_success**: Number of additional frames to record after a success (reward=1) is detected
- **fps**: Number of frames per second to record
- **push_to_hub**: Whether to push the dataset to the hub

The `number_of_steps_after_success` parameter is crucial as it allows you to collect more positive examples. When a success is detected, the system will continue recording for the specified number of steps while maintaining the reward=1 label. Otherwise, there won't be enough states in the dataset labeled to 1 to train a good classifier.

Example configuration section for data collection:

```json
{
    "mode": "record",
    "repo_id": "hf_username/dataset_name",
    "dataset_root": "data/your_dataset",
    "num_episodes": 20,
    "push_to_hub": true,
    "fps": 10,
    "number_of_steps_after_success": 15
}
```

## 2. Reward Classifier Configuration

The reward classifier is configured using `configuration_classifier.py`. Here are the key parameters:

- **model_name**: Base model architecture (e.g., we mainly use "helper2424/resnet10")
- **model_type**: "cnn" or "transformer"
- **num_cameras**: Number of camera inputs
- **num_classes**: Number of output classes (typically 2 for binary success/failure)
- **hidden_dim**: Size of hidden representation
- **dropout_rate**: Regularization parameter
- **learning_rate**: Learning rate for optimizer

Example configuration from `reward_classifier_train_config.json`:

```json
{
  "policy": {
    "type": "reward_classifier",
    "model_name": "helper2424/resnet10",
    "model_type": "cnn",
    "num_cameras": 2,
    "num_classes": 2,
    "hidden_dim": 256,
    "dropout_rate": 0.1,
    "learning_rate": 1e-4,
    "device": "cuda",
    "use_amp": true,
    "input_features": {
      "observation.images.front": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      },
      "observation.images.side": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      }
    }
  }
}
```

## 3. Training the Classifier

To train the classifier, use the `train.py` script with your configuration:

```bash
python lerobot/scripts/train.py --config_path lerobot/configs/reward_classifier_train_config.json
```

## 4. Deploying and Testing the Model

To use your trained reward classifier, configure the `HILSerlRobotEnvConfig` to use your model:

```python
env_config = HILSerlRobotEnvConfig(
    reward_classifier_pretrained_path="path_to_your_pretrained_trained_model",
    # Other environment parameters
)
```
or set the argument in the json config file.

```json
{
    "reward_classifier_pretrained_path": "path_to_your_pretrained_model"
}
```

Run gym_manipulator.py to test the model.
```bash
python lerobot/scripts/server/gym_manipulator.py --config_path lerobot/configs/env_config.json
```

The reward classifier will automatically provide rewards based on the visual input from the robot's cameras.

## Example Workflow

1. **Create the configuration files**:
   Create the necessary json configuration files for the reward classifier and the environment. Check the `json_examples` directory for examples.
2. **Collect a dataset**:
   ```bash
   python lerobot/scripts/server/gym_manipulator.py --config_path lerobot/configs/env_config.json
   ```
3. **Train the classifier**:
   ```bash
   python lerobot/scripts/train.py --config_path lerobot/configs/reward_classifier_train_config.json
   ```
4. **Test the classifier**:
   ```bash
   python lerobot/scripts/server/gym_manipulator.py --config_path lerobot/configs/env_config.json
   # private usage
   PYTHONPATH=/home/zekaijin/lerobot-hilserl/lerobot/src python lerobot/src/lerobot/scripts/rl/gym_manipulator.py --config_path configs/gym_hil_env.json
   ```

## Why Use Reward Classifiers Instead of Manual Success Detection?
**Brief Summary**: In real robot training, using "manual success button press = 1, everything else = 0" sparse rewards is severely compromised by human reaction delays and triggering errors. The reward classifier transforms this high-latency, low-consistency signal into "per-frame, low-latency, reusable" success detection.

### Key Benefits (in the context of HIL-SERL):
#### 1. Eliminates Misaligned Samples from Human Delay
- **Problem**: Manual button presses typically have 0.3–1.0s delay; at 10 Hz sampling, this creates 3–10 frames of "successful but unrewarded" false negative samples
- **Impact**: Positive samples become severely diluted, making it difficult for even binary classifiers to learn stably, let alone RL algorithms
- **Solution**: A trained success/failure binary classifier (based on images + proprioception) can provide instant success detection at control frequency, significantly reducing misalignment

#### 2. Unified and Reusable Reward Source
- Once trained, the classifier serves as sparse reward during online training: success → 1, otherwise → 0
- Automatically terminates episodes (done=true), saving robot time and wear while reducing human annotation burden
- Provides consistent reward signals across different training sessions

#### 3. Offline Relabeling and Data Cleaning
- Collected trajectories can be relabeled frame-by-frame using the same classifier
- Removes false negative samples caused by human delay, improving data purity in the experience replay buffer
- This capability is impossible with manual terminal labeling

#### 4. Better Generalization and Consistency
- Manual threshold/command-based "success" definitions are often inconsistent across subtle variations, lighting changes, and different operators
- Classifiers can maintain stability across different sessions/operators through data augmentation and validation set threshold calibration
- **HIL-SERL workflow**: Collect positive/negative samples via teleoperation → train binary reward classifier → add few demonstrations to Demo Buffer → use classifier rewards + minimal human intervention during online RL, with intervention frequency decreasing as performance improves

#### 5. Engineering Efficiency
Unlike relying on manual "final button press" for each episode, the classifier enables long-term unattended data collection once available. This approach is a "first-class citizen" in the SERL/HIL-SERL codebase, ready to use out of the box

### Practical Impact
The reward classifier transforms the learning process from a high-latency, error-prone manual system to a robust, automated, and scalable solution that significantly improves sample efficiency and training stability in real robot scenarios.

### Results
After 80,000 training steps and manual intervention (approximately 1 hour), our policy achieved a 100% success rate in the pick_cube_sim environment. Below are our training curves and the final policy results.



## ACFQL (action chunking flowing Q-learning)
### Run tests
   ```bash
   # 测试 ACFQL processor
  cd /home/zekaijin/lerobot-hilserl-ufactory/lerobot && PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src /home/zekaijin/miniconda3/
  envs/lerobot/bin/python -m pytest -sv ./tests/processor/test_acfql_processor.py
   # 测试 observation processor（单个图像）
  cd /home/zekaijin/lerobot-hilserl-ufactory/lerobot && PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src /home/zekaijin/miniconda3/envs/lerobot/bin/python -m pytest -sv ./tests/processor/test_observation_processor.py::test_process_single_image
  # 测试 replay buffer n-steps
  cd /home/zekaijin/lerobot-hilserl-ufactory/lerobot && PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src /home/zekaijin/miniconda3/envs/lerobot/bin/python -m pytest -sv ./tests/rl/acfql/test_replay_buffer_n_steps.py
  # 第二个命令中的测试函数名是 test_process_single_image（不是 test_process_single_image_cuda）。如果需要测试 CUDA 相关功能，可以运行整个文件：
  cd /home/zekaijin/lerobot-hilserl-ufactory/lerobot && PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src /home/zekaijin/miniconda3/envs/lerobot/bin/python -m pytest -sv ./tests/processor/test_observation_processor.py
   ```
  
## 人工采集数据 ac-fql gamepad 
  ```bash
  # pick and lift env without ft
  PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.rl.acfql.gym_manipulator   --config_path configs/simulation/acfql/pick_and_lift/gym_hil_env_fql.json
  # pick and lift env with ft
  PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src python -m lerobot.rl.acfql.gym_manipulator   --config_path configs/simulation/acfql/pick_and_lift/gym_hil_env_fql_ft.json
  ```

## train a policy on PandaPickCubeGamepad-v0 sim
This command will launch the learner to do 4k offline steps, then will wait for receiving 4k transitions from actor, and will start doing online RL.
   ```bash
   # first start learner without ft
      cd /home/zekaijin/lerobot-hilserl-ufactory/lerobot && \
      PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src \
      /home/zekaijin/miniconda3/envs/lerobot/bin/python -m lerobot.rl.acfql.learner \
        --config_path=../configs/simulation/acfql/pick_and_lift/train_gym_hil_env_fql.json \
        --policy.offline_steps=4000 \
        --policy.online_step_before_learning=4000
    ```

This command launch the actor, it will wait for the model paramaters, and then will start getting transitions
  ```bash
  # then start actor without ft
  cd /home/zekaijin/lerobot-hilserl-ufactory/lerobot && \
  PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src \
  /home/zekaijin/miniconda3/envs/lerobot/bin/python -m lerobot.rl.acfql.actor \
    --config_path=../configs/simulation/acfql/pick_and_lift/train_gym_hil_env_fql.json \
    --policy.offline_steps=4000
  ```

In case the actor do not start getting transitions, and the learner wait for the actor's transitions. -> resume the actor from the learner checkpoint with offline_steps=0. The actor will start getting transitions right after loading the model from the checkpoint. Once the learner receives enough transitions, it will start online RL, and also will start sending model params to the actor.
   
  ```bash
  # Learner（恢复训练）
  cd /home/zekaijin/lerobot-hilserl-ufactory/lerobot && \
  PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src \
  /home/zekaijin/miniconda3/envs/lerobot/bin/python -m lerobot.rl.acfql.learner \
    --config_path=outputs/train/2025-12-16/11-18-35_franka_sim_pick_lift_fql/checkpoints/last/pretrained_model/train_config.json \
    --resume=true

  # 如果 Actor 无法开始收集数据（恢复训练）
  cd /home/zekaijin/lerobot-hilserl-ufactory/lerobot && \
  PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src \
  /home/zekaijin/miniconda3/envs/lerobot/bin/python -m lerobot.rl.acfql.actor \
    --config_path=outputs/train/2025-12-16/11-18-35_franka_sim_pick_lift_fql/checkpoints/last/pretrained_model/train_config.json \
    --resume=true \
    --policy.offline_steps=0
  ```

## Training PICK AND LIFT ENV pipeline with ft sensor
  ```bash
  # learner
  cd /home/zekaijin/lerobot-hilserl-ufactory/lerobot && \
  PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src \
      /home/zekaijin/miniconda3/envs/lerobot/bin/python -m lerobot.rl.acfql.learner \
        --config_path=../configs/simulation/acfql/pick_and_lift/train_gym_hil_env_fql_ft.json \
        --policy.offline_steps=4000 \
        --policy.online_step_before_learning=4000

  # actor
  cd /home/zekaijin/lerobot-hilserl-ufactory/lerobot && \
  PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src \
  /home/zekaijin/miniconda3/envs/lerobot/bin/python -m lerobot.rl.acfql.actor \
    --config_path=../configs/simulation/acfql/pick_and_lift/train_gym_hil_env_fql_ft.json \
    --policy.offline_steps=4000

  # 如果 Actor 无法开始收集数据（恢复训练）
  cd /home/zekaijin/lerobot-hilserl-ufactory/lerobot && \
  PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src \
  /home/zekaijin/miniconda3/envs/lerobot/bin/python -m lerobot.rl.acfql.actor \
    --config_path=./outputs/.../checkpoints/last/pretrained_model/train_gym_hil_env_fql_ft.json \
    --resume=true \
    --policy.offline_steps=0
  ```

## Evaluation
  ```bash
  # Example Usage
  cd /home/zekaijin/lerobot-hilserl-ufactory/lerobot && \
  PYTHONPATH=/home/zekaijin/lerobot-hilserl-ufactory/lerobot/src \
  /home/zekaijin/miniconda3/envs/lerobot/bin/python -m lerobot.rl.acfql.eval_policy \
    --config_path=../configs/simulation/acfql/train_gym_hil_env_fql.json \
    --policy.pretrained_path=outputs/train/2025-12-16/11-18-35_franka_sim_pick_lift_fql/checkpoints/last/pretrained_model \
    --eval.n_episodes=50
    ```