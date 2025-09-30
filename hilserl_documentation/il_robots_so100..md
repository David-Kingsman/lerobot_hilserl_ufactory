# Imitation Learning on Real-World Robots

This tutorial will explain how to train a neural network to control a real robot autonomously.

**You'll learn:**

1. How to record and visualize your dataset.
2. How to train a policy using your data and prepare it for evaluation.
3. How to evaluate your policy and visualize the results.

By following these steps, you'll be able to replicate tasks, such as picking up a Lego block and placing it in a bin with a high success rate, as shown in the video below.

<details>
<summary><strong>Video: pickup lego block task</strong></summary>

<div class="video-container">
  <video controls width="600">
    <source
      src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/lerobot_task.mp4"
      type="video/mp4"
    />
  </video>
</div>

</details>

This tutorial isn’t tied to a specific robot: we walk you through the commands and API snippets you can adapt for any supported platform.

During data collection, you’ll use a “teloperation” device, such as a leader arm or keyboard to teleoperate the robot and record its motion trajectories.

Once you’ve gathered enough trajectories, you’ll train a neural network to imitate these trajectories and deploy the trained model so your robot can perform the task autonomously.

If you run into any issues at any point, jump into our [Discord community](https://discord.com/invite/s3KuuzsPFb) for support.

## Set up and Calibrate

If you haven't yet set up and calibrated your robot and teleop device, please do so by following the robot-specific tutorial.

## Teleoperate

In this example, we’ll demonstrate how to teleoperate the SO101 robot. For each command, we also provide a corresponding API example.

Note that the `id` associated with a robot is used to store the calibration file. It's important to use the same `id` when teleoperating, recording, and evaluating when using the same setup.

<hfoptions id="teleoperate_so101">
<hfoption id="Command">
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=my_awesome_leader_arm
```
</hfoption>
<hfoption id="API example">

<!-- prettier-ignore-start -->
```python
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

robot_config = SO101FollowerConfig(
    port="/dev/tty.usbmodem58760431541",
    id="my_red_robot_arm",
)

teleop_config = SO101LeaderConfig(
    port="/dev/tty.usbmodem58760431551",
    id="my_blue_leader_arm",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    action = teleop_device.get_action()
    robot.send_action(action)
```
<!-- prettier-ignore-end -->

</hfoption>
</hfoptions>

The teleoperate command will automatically:

1. Identify any missing calibrations and initiate the calibration procedure.
2. Connect the robot and teleop device and start teleoperation.

## Cameras

To add cameras to your setup, follow this [Guide](./cameras#setup-cameras).

## Teleoperate with cameras

With `rerun`, you can teleoperate again while simultaneously visualizing the camera feeds and joint positions. In this example, we’re using the Koch arm.

<hfoptions id="teleoperate_koch_camera">
<hfoption id="Command">
```bash
lerobot-teleoperate \
    --robot.type=koch_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --teleop.type=koch_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true
```
</hfoption>
<hfoption id="API example">

<!-- prettier-ignore-start -->
```python
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.koch_leader import KochLeaderConfig, KochLeader
from lerobot.robots.koch_follower import KochFollowerConfig, KochFollower

camera_config = {
    "front": OpenCVCameraConfig(index_or_path=0, width=1920, height=1080, fps=30)
}

robot_config = KochFollowerConfig(
    port="/dev/tty.usbmodem585A0076841",
    id="my_red_robot_arm",
    cameras=camera_config
)

teleop_config = KochLeaderConfig(
    port="/dev/tty.usbmodem58760431551",
    id="my_blue_leader_arm",
)

robot = KochFollower(robot_config)
teleop_device = KochLeader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    observation = robot.get_observation()
    action = teleop_device.get_action()
    robot.send_action(action)
```
<!-- prettier-ignore-end -->

</hfoption>
</hfoptions>

## Record a dataset

Once you're familiar with teleoperation, you can record your first dataset.

We use the Hugging Face hub features for uploading your dataset. If you haven't previously used the Hub, make sure you can login via the cli using a write-access token, this token can be generated from the [Hugging Face settings](https://huggingface.co/settings/tokens).

Add your token to the CLI by running this command:

```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

Then store your Hugging Face repository name in a variable:

```bash
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
```

Now you can record a dataset. To record 5 episodes and upload your dataset to the hub, adapt the code below for your robot and execute the command or API example.

<hfoptions id="record">
<hfoption id="Command">
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem585A0076841 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.num_episodes=5 \
    --dataset.single_task="Grab the black cube"
```
</hfoption>
<hfoption id="API example">

<!-- prettier-ignore-start -->
```python
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.record import record_loop

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "My task description"

# Create the robot and teleoperator configurations
camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem58760434471", id="my_awesome_follower_arm", cameras=camera_config
)
teleop_config = SO100LeaderConfig(port="/dev/tty.usbmodem585A0077581", id="my_awesome_leader_arm")

# Initialize the robot and teleoperator
robot = SO100Follower(robot_config)
teleop = SO100Leader(teleop_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="<hf_username>/<dataset_repo_id>",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
_init_rerun(session_name="recording")

# Connect the robot and teleoperator
robot.connect()
teleop.connect()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
robot.disconnect()
teleop.disconnect()
dataset.push_to_hub()
```
<!-- prettier-ignore-end -->

</hfoption>
</hfoptions>

#### Dataset upload

Locally, your dataset is stored in this folder: `~/.cache/huggingface/lerobot/{repo-id}`. At the end of data recording, your dataset will be uploaded on your Hugging Face page (e.g. `https://huggingface.co/datasets/${HF_USER}/so101_test`) that you can obtain by running:

```bash
echo https://huggingface.co/datasets/${HF_USER}/so101_test
```

Your dataset will be automatically tagged with `LeRobot` for the community to find it easily, and you can also add custom tags (in this case `tutorial` for example).

You can look for other LeRobot datasets on the hub by searching for `LeRobot` [tags](https://huggingface.co/datasets?other=LeRobot).

You can also push your local dataset to the Hub manually, running:

```bash
huggingface-cli upload ${HF_USER}/record-test ~/.cache/huggingface/lerobot/{repo-id} --repo-type dataset
```

#### Record function

The `record` function provides a suite of tools for capturing and managing data during robot operation:

##### 1. Data Storage

- Data is stored using the `LeRobotDataset` format and is stored on disk during recording.
- By default, the dataset is pushed to your Hugging Face page after recording.
  - To disable uploading, use `--dataset.push_to_hub=False`.

##### 2. Checkpointing and Resuming

- Checkpoints are automatically created during recording.
- If an issue occurs, you can resume by re-running the same command with `--resume=true`. When resuming a recording, `--dataset.num_episodes` must be set to the **number of additional episodes to be recorded**, and not to the targeted total number of episodes in the dataset !
- To start recording from scratch, **manually delete** the dataset directory.

##### 3. Recording Parameters

Set the flow of data recording using command-line arguments:

- `--dataset.episode_time_s=60`
  Duration of each data recording episode (default: **60 seconds**).
- `--dataset.reset_time_s=60`
  Duration for resetting the environment after each episode (default: **60 seconds**).
- `--dataset.num_episodes=50`
  Total number of episodes to record (default: **50**).

##### 4. Keyboard Controls During Recording

Control the data recording flow using keyboard shortcuts:

- Press **Right Arrow (`→`)**: Early stop the current episode or reset time and move to the next.
- Press **Left Arrow (`←`)**: Cancel the current episode and re-record it.
- Press **Escape (`ESC`)**: Immediately stop the session, encode videos, and upload the dataset.

#### Tips for gathering data

Once you're comfortable with data recording, you can create a larger dataset for training. A good starting task is grasping an object at different locations and placing it in a bin. We suggest recording at least 50 episodes, with 10 episodes per location. Keep the cameras fixed and maintain consistent grasping behavior throughout the recordings. Also make sure the object you are manipulating is visible on the camera's. A good rule of thumb is you should be able to do the task yourself by only looking at the camera images.

In the following sections, you’ll train your neural network. After achieving reliable grasping performance, you can start introducing more variations during data collection, such as additional grasp locations, different grasping techniques, and altering camera positions.

Avoid adding too much variation too quickly, as it may hinder your results.

If you want to dive deeper into this important topic, you can check out the [blog post](https://huggingface.co/blog/lerobot-datasets#what-makes-a-good-dataset) we wrote on what makes a good dataset.

#### Troubleshooting:

- On Linux, if the left and right arrow keys and escape key don't have any effect during data recording, make sure you've set the `$DISPLAY` environment variable. See [pynput limitations](https://pynput.readthedocs.io/en/latest/limitations.html#linux).

## Visualize a dataset

If you uploaded your dataset to the hub with `--control.push_to_hub=true`, you can [visualize your dataset online](https://huggingface.co/spaces/lerobot/visualize_dataset) by copy pasting your repo id given by:

```bash
echo ${HF_USER}/so101_test
```

## Replay an episode

A useful feature is the `replay` function, which allows you to replay any episode that you've recorded or episodes from any dataset out there. This function helps you test the repeatability of your robot's actions and assess transferability across robots of the same model.

You can replay the first episode on your robot with either the command below or with the API example:

<hfoptions id="replay">
<hfoption id="Command">
```bash
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=my_awesome_follower_arm \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.episode=0 # choose the episode you want to replay
```
</hfoption>
<hfoption id="API example">

<!-- prettier-ignore-start -->
```python
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

episode_idx = 0

robot_config = SO100FollowerConfig(port="/dev/tty.usbmodem58760434471", id="my_awesome_follower_arm")

robot = SO100Follower(robot_config)
robot.connect()

dataset = LeRobotDataset("<hf_username>/<dataset_repo_id>", episodes=[episode_idx])
actions = dataset.hf_dataset.select_columns("action")

log_say(f"Replaying episode {episode_idx}")
for idx in range(dataset.num_frames):
    t0 = time.perf_counter()

    action = {
        name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
    }
    robot.send_action(action)

    busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))

robot.disconnect()
```
<!-- prettier-ignore-end -->

</hfoption>
</hfoptions>

Your robot should replicate movements similar to those you recorded. For example, check out [this video](https://x.com/RemiCadene/status/1793654950905680090) where we use `replay` on a Aloha robot from [Trossen Robotics](https://www.trossenrobotics.com).

## Train a policy

To train a policy to control your robot, use the [`lerobot-train`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/train.py) script. A few arguments are required. Here is an example command:

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/so101_test \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/my_policy
```

Let's explain the command:

1. We provided the dataset as argument with `--dataset.repo_id=${HF_USER}/so101_test`.
2. We provided the policy with `policy.type=act`. This loads configurations from [`configuration_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py). Importantly, this policy will automatically adapt to the number of motor states, motor actions and cameras of your robot (e.g. `laptop` and `phone`) which have been saved in your dataset.
3. We provided `policy.device=cuda` since we are training on a Nvidia GPU, but you could use `policy.device=mps` to train on Apple silicon.
4. We provided `wandb.enable=true` to use [Weights and Biases](https://docs.wandb.ai/quickstart) for visualizing training plots. This is optional but if you use it, make sure you are logged in by running `wandb login`.

Training should take several hours. You will find checkpoints in `outputs/train/act_so101_test/checkpoints`.

To resume training from a checkpoint, below is an example command to resume from `last` checkpoint of the `act_so101_test` policy:

```bash
lerobot-train \
  --config_path=outputs/train/act_so101_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

If you do not want to push your model to the hub after training use `--policy.push_to_hub=false`.

Additionally you can provide extra `tags` or specify a `license` for your model or make the model repo `private` by adding this: `--policy.private=true --policy.tags=\[ppo,rl\] --policy.license=mit`

#### Train using Google Colab

If your local computer doesn't have a powerful GPU you could utilize Google Colab to train your model by following the [ACT training notebook](./notebooks#training-act).

#### Upload policy checkpoints

Once training is done, upload the latest checkpoint with:

```bash
huggingface-cli upload ${HF_USER}/act_so101_test \
  outputs/train/act_so101_test/checkpoints/last/pretrained_model
```

You can also upload intermediate checkpoints with:

```bash
CKPT=010000
huggingface-cli upload ${HF_USER}/act_so101_test${CKPT} \
  outputs/train/act_so101_test/checkpoints/${CKPT}/pretrained_model
```

## Run inference and evaluate your policy

You can use the `record` script from [`lerobot/record.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/record.py) with a policy checkpoint as input, to run inference and evaluate your policy. For instance, run this command or API example to run inference and record 10 evaluation episodes:

<hfoptions id="eval">
<hfoption id="Command">
```bash
lerobot-record  \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.cameras="{ up: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, side: {type: intelrealsense, serial_number_or_name: 233522074606, width: 640, height: 480, fps: 30}}" \
  --robot.id=my_awesome_follower_arm \
  --display_data=false \
  --dataset.repo_id=${HF_USER}/eval_so100 \
  --dataset.single_task="Put lego brick into the transparent box" \
  # <- Teleop optional if you want to teleoperate in between episodes \
  # --teleop.type=so100_leader \
  # --teleop.port=/dev/ttyACM0 \
  # --teleop.id=my_awesome_leader_arm \
  --policy.path=${HF_USER}/my_policy
```
</hfoption>
<hfoption id="API example">

<!-- prettier-ignore-start -->
```python
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.record import record_loop
from lerobot.policies.factory import make_processor

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "My task description"
HF_MODEL_ID = "<hf_username>/<model_repo_id>"
HF_DATASET_ID = "<hf_username>/<eval_dataset_repo_id>"

# Create the robot configuration
camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem58760434471", id="my_awesome_follower_arm", cameras=camera_config
)

# Initialize the robot
robot = SO100Follower(robot_config)

# Initialize the policy
policy = ACTPolicy.from_pretrained(HF_MODEL_ID)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=HF_DATASET_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
_init_rerun(session_name="recording")

# Connect the robot
robot.connect()

preprocessor, postprocessor = make_processor(
    policy_cfg=policy,
    pretrained_path=HF_MODEL_ID,
    dataset_stats=dataset.meta.stats,
)

for episode_idx in range(NUM_EPISODES):
    log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")

    # Run the policy inference loop
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    dataset.save_episode()

# Clean up
robot.disconnect()
dataset.push_to_hub()
```
<!-- prettier-ignore-end -->

</hfoption>
</hfoptions>

As you can see, it's almost the same command as previously used to record your training dataset. Two things changed:

1. There is an additional `--control.policy.path` argument which indicates the path to your policy checkpoint with (e.g. `outputs/train/eval_act_so101_test/checkpoints/last/pretrained_model`). You can also use the model repository if you uploaded a model checkpoint to the hub (e.g. `${HF_USER}/act_so101_test`).
2. The name of dataset begins by `eval` to reflect that you are running inference (e.g. `${HF_USER}/eval_act_so101_test`).



# SO-100

In the steps below, we explain how to assemble the SO-100 robot.

## Source the parts

Follow this [README](https://github.com/TheRobotStudio/SO-ARM100/blob/main/SO100.md). It contains the bill of materials, with a link to source the parts, as well as the instructions to 3D print the parts. And advise if it's your first time printing or if you don't own a 3D printer.

## Install LeRobot 🤗

To install LeRobot, follow our [Installation Guide](./installation)

In addition to these instructions, you need to install the Feetech SDK:

```bash
pip install -e ".[feetech]"
```

## Configure the motors

**Note:**
Unlike the SO-101, the motor connectors are not easily accessible once the arm is assembled, so the configuration step must be done beforehand.

### 1. Find the USB ports associated with each arm

To find the port for each bus servo adapter, run this script:

```bash
lerobot-find-port
```

<hfoptions id="example">
<hfoption id="Mac">

Example output:

```
Finding all available ports for the MotorBus.
['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
Remove the USB cable from your MotorsBus and press Enter when done.

[...Disconnect corresponding leader or follower arm and press Enter...]

The port of this MotorsBus is /dev/tty.usbmodem575E0032081
Reconnect the USB cable.
```

Where the found port is: `/dev/tty.usbmodem575E0032081` corresponding to your leader or follower arm.

</hfoption>
<hfoption id="Linux">

On Linux, you might need to give access to the USB ports by running:

```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

Example output:

```
Finding all available ports for the MotorBus.
['/dev/ttyACM0', '/dev/ttyACM1']
Remove the usb cable from your MotorsBus and press Enter when done.

[...Disconnect corresponding leader or follower arm and press Enter...]

The port of this MotorsBus is /dev/ttyACM1
Reconnect the USB cable.
```

Where the found port is: `/dev/ttyACM1` corresponding to your leader or follower arm.

</hfoption>
</hfoptions>

### 2. Set the motors ids and baudrates

Each motor is identified by a unique id on the bus. When brand new, motors usually come with a default id of `1`. For the communication to work properly between the motors and the controller, we first need to set a unique, different id to each motor. Additionally, the speed at which data is transmitted on the bus is determined by the baudrate. In order to talk to each other, the controller and all the motors need to be configured with the same baudrate.

To that end, we first need to connect to each motor individually with the controller in order to set these. Since we will write these parameters in the non-volatile section of the motors' internal memory (EEPROM), we'll only need to do this once.

If you are repurposing motors from another robot, you will probably also need to perform this step as the ids and baudrate likely won't match.

#### Follower

Connect the usb cable from your computer and the power supply to the follower arm's controller board. Then, run the following command or run the API example with the port you got from the previous step. You'll also need to give your leader arm a name with the `id` parameter.

For a visual reference on how to set the motor ids please refer to [this video](https://huggingface.co/docs/lerobot/en/so101#setup-motors-video) where we follow the process for the SO101 arm.

<hfoptions id="setup_motors">
<hfoption id="Command">

```bash
lerobot-setup-motors \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem585A0076841  # <- paste here the port found at previous step
```

</hfoption>
<hfoption id="API example">

<!-- prettier-ignore-start -->
```python
from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig

config = SO100FollowerConfig(
    port="/dev/tty.usbmodem585A0076841",
    id="my_awesome_follower_arm",
)
follower = SO100Follower(config)
follower.setup_motors()
```
<!-- prettier-ignore-end -->

</hfoption>
</hfoptions>

You should see the following instruction

```
Connect the controller board to the 'gripper' motor only and press enter.
```

As instructed, plug the gripper's motor. Make sure it's the only motor connected to the board, and that the motor itself is not yet daisy-chained to any other motor. As you press `[Enter]`, the script will automatically set the id and baudrate for that motor.

<details>
<summary>Troubleshooting</summary>

If you get an error at that point, check your cables and make sure they are plugged in properly:

<ul>
  <li>Power supply</li>
  <li>USB cable between your computer and the controller board</li>
  <li>The 3-pin cable from the controller board to the motor</li>
</ul>

If you are using a Waveshare controller board, make sure that the two jumpers are set on the `B` channel (USB).

</details>

You should then see the following message:

```
'gripper' motor id set to 6
```

Followed by the next instruction:

```
Connect the controller board to the 'wrist_roll' motor only and press enter.
```

You can disconnect the 3-pin cable from the controller board, but you can leave it connected to the gripper motor on the other end, as it will already be in the right place. Now, plug in another 3-pin cable to the wrist roll motor and connect it to the controller board. As with the previous motor, make sure it is the only motor connected to the board and that the motor itself isn't connected to any other one.

Repeat the operation for each motor as instructed.

> [!TIP]
> Check your cabling at each step before pressing Enter. For instance, the power supply cable might disconnect as you manipulate the board.

When you are done, the script will simply finish, at which point the motors are ready to be used. You can now plug the 3-pin cable from each motor to the next one, and the cable from the first motor (the 'shoulder pan' with id=1) to the controller board, which can now be attached to the base of the arm.

#### Leader

Do the same steps for the leader arm.

<hfoptions id="setup_motors">
<hfoption id="Command">
```bash
lerobot-setup-motors \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem575E0031751  # <- paste here the port found at previous step
```
</hfoption>
<hfoption id="API example">

<!-- prettier-ignore-start -->
```python
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig

config = SO100LeaderConfig(
    port="/dev/tty.usbmodem585A0076841",
    id="my_awesome_leader_arm",
)
leader = SO100Leader(config)
leader.setup_motors()
```
<!-- prettier-ignore-end -->

</hfoption>
</hfoptions>

## Step-by-Step Assembly Instructions

## Remove the gears of the 6 leader motors

<details>
<summary><strong>Video removing gears</strong></summary>

<div class="video-container">
  <video controls width="600">
    <source
      src="https://github.com/user-attachments/assets/0c95b88c-5b85-413d-ba19-aee2f864f2a7"
      type="video/mp4"
    />
  </video>
</div>

</details>

Follow the video for removing gears. You need to remove the gear for the motors of the leader arm. As a result, you will only use the position encoding of the motor and reduce friction to more easily operate the leader arm.

### Clean Parts

Remove all support material from the 3D-printed parts. The easiest way to do this is using a small screwdriver to get underneath the support material.

### Additional Guidance

<details>
<summary><strong>Video assembling arms</strong></summary>

<div class="video-container">
  <video controls width="600">
    <source
      src="https://github.com/user-attachments/assets/488a39de-0189-4461-9de3-05b015f90cca"
      type="video/mp4"
    />
  </video>
</div>

</details>

**Note:**
This video provides visual guidance for assembling the arms, but it doesn't specify when or how to do the wiring. Inserting the cables beforehand is much easier than doing it afterward. The first arm may take a bit more than 1 hour to assemble, but once you get used to it, you can assemble the second arm in under 1 hour.

---

### First Motor

**Step 2: Insert Wires**

- Insert two wires into the first motor.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_1.webp"
  style="height:300px;"
/>

**Step 3: Install in Base**

- Place the first motor into the base.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_2.webp"
  style="height:300px;"
/>

**Step 4: Secure Motor**

- Fasten the motor with 4 screws. Two from the bottom and two from top.

**Step 5: Attach Motor Holder**

- Slide over the first motor holder and fasten it using two screws (one on each side).

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_4.webp"
  style="height:300px;"
/>

**Step 6: Attach Motor Horns**

- Install both motor horns, securing the top horn with a screw. Try not to move the motor position when attaching the motor horn, especially for the leader arms, where we removed the gears.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_5.webp"
  style="height:300px;"
/>

<details>
  <summary>
    <strong>Video adding motor horn</strong>
  </summary>
  <video src="https://github.com/user-attachments/assets/ef3391a4-ad05-4100-b2bd-1699bf86c969"></video>
</details>

**Step 7: Attach Shoulder Part**

- Route one wire to the back of the robot and the other to the left or towards you (see photo).
- Attach the shoulder part.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_6.webp"
  style="height:300px;"
/>

**Step 8: Secure Shoulder**

- Tighten the shoulder part with 4 screws on top and 4 on the bottom
  _(access bottom holes by turning the shoulder)._

---

### Second Motor Assembly

**Step 9: Install Motor 2**

- Slide the second motor in from the top and link the wire from motor 1 to motor 2.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_8.webp"
  style="height:300px;"
/>

**Step 10: Attach Shoulder Holder**

- Add the shoulder motor holder.
- Ensure the wire from motor 1 to motor 2 goes behind the holder while the other wire is routed upward (see photo).
- This part can be tight to assemble, you can use a workbench like the image or a similar setup to push the part around the motor.

<div style="display: flex;">
  <img
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_9.webp"
    style="height:250px;"
  />
  <img
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_10.webp"
    style="height:250px;"
  />
  <img
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_12.webp"
    style="height:250px;"
  />
</div>

**Step 11: Secure Motor 2**

- Fasten the second motor with 4 screws.

**Step 12: Attach Motor Horn**

- Attach both motor horns to motor 2, again use the horn screw.

**Step 13: Attach Base**

- Install the base attachment using 2 screws.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_11.webp" style="height:300px;">

**Step 14: Attach Upper Arm**

- Attach the upper arm with 4 screws on each side.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_13.webp" style="height:300px;">

---

### Third Motor Assembly

**Step 15: Install Motor 3**

- Route the motor cable from motor 2 through the cable holder to motor 3, then secure motor 3 with 4 screws.

**Step 16: Attach Motor Horn**

- Attach both motor horns to motor 3 and secure one again with a horn screw.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_14.webp"
  style="height:300px;"
/>

**Step 17: Attach Forearm**

- Connect the forearm to motor 3 using 4 screws on each side.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_15.webp"
  style="height:300px;"
/>

---

### Fourth Motor Assembly

**Step 18: Install Motor 4**

- Slide in motor 4, attach the cable from motor 3, and secure the cable in its holder with a screw.

<div style="display: flex;">
  <img
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_16.webp"
    style="height:300px;"
  />
  <img
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_19.webp"
    style="height:300px;"
  />
</div>

**Step 19: Attach Motor Holder 4**

- Install the fourth motor holder (a tight fit). Ensure one wire is routed upward and the wire from motor 3 is routed downward (see photo).

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_17.webp"
  style="height:300px;"
/>

**Step 20: Secure Motor 4 & Attach Horn**

- Fasten motor 4 with 4 screws and attach its motor horns, use for one a horn screw.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_18.webp"
  style="height:300px;"
/>

---

### Wrist Assembly

**Step 21: Install Motor 5**

- Insert motor 5 into the wrist holder and secure it with 2 front screws.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_20.webp"
  style="height:300px;"
/>

**Step 22: Attach Wrist**

- Connect the wire from motor 4 to motor 5. And already insert the other wire for the gripper.
- Secure the wrist to motor 4 using 4 screws on both sides.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_22.webp"
  style="height:300px;"
/>

**Step 23: Attach Wrist Horn**

- Install only one motor horn on the wrist motor and secure it with a horn screw.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_23.webp"
  style="height:300px;"
/>

---

### Follower Configuration

**Step 24: Attach Gripper**

- Attach the gripper to motor 5.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_24.webp"
  style="height:300px;"
/>

**Step 25: Install Gripper Motor**

- Insert the gripper motor, connect the motor wire from motor 5 to motor 6, and secure it with 3 screws on each side.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_25.webp"
  style="height:300px;"
/>

**Step 26: Attach Gripper Horn & Claw**

- Attach the motor horns and again use a horn screw.
- Install the gripper claw and secure it with 4 screws on both sides.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_26.webp"
  style="height:300px;"
/>

**Step 27: Mount Controller**

- Attach the motor controller to the back of the robot.

<div style="display: flex;">
  <img
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_27.webp"
    style="height:300px;"
  />
  <img
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_28.webp"
    style="height:300px;"
  />
</div>

_Assembly complete – proceed to Leader arm assembly._

---

### Leader Configuration

For the leader configuration, perform **Steps 1–23**. Make sure that you removed the motor gears from the motors.

**Step 24: Attach Leader Holder**

- Mount the leader holder onto the wrist and secure it with a screw.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_29.webp"
  style="height:300px;"
/>

**Step 25: Attach Handle**

- Attach the handle to motor 5 using 4 screws.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_30.webp"
  style="height:300px;"
/>

**Step 26: Install Gripper Motor**

- Insert the gripper motor, secure it with 3 screws on each side, attach a motor horn using a horn screw, and connect the motor wire.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_31.webp"
  style="height:300px;"
/>

**Step 27: Attach Trigger**

- Attach the follower trigger with 4 screws.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_32.webp"
  style="height:300px;"
/>

**Step 28: Mount Controller**

- Attach the motor controller to the back of the robot.

<div style="display: flex;">
  <img
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_27.webp"
    style="height:300px;"
  />
  <img
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100_assembly_28.webp"
    style="height:300px;"
  />
</div>

## Calibrate

Next, you'll need to calibrate your robot to ensure that the leader and follower arms have the same position values when they are in the same physical position.
The calibration process is very important because it allows a neural network trained on one robot to work on another.

#### Follower

Run the following command or API example to calibrate the follower arm:

<hfoptions id="calibrate_follower">
<hfoption id="Command">

```bash
lerobot-calibrate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm 
```

</hfoption>
<hfoption id="API example">

<!-- prettier-ignore-start -->
```python
from lerobot.robots.so100_follower import SO100FollowerConfig, SO100Follower

config = SO100FollowerConfig(
    port="/dev/tty.usbmodem585A0076891",
    id="my_awesome_follower_arm",
)

follower = SO100Follower(config)
follower.connect(calibrate=False)
follower.calibrate()
follower.disconnect()
```
<!-- prettier-ignore-end -->

</hfoption>
</hfoptions>

We unified the calibration method for most robots. Thus, the calibration steps for this SO100 arm are the same as the steps for the Koch and SO101. First, we have to move the robot to the position where each joint is in the middle of its range, then we press `Enter`. Secondly, we move all joints through their full range of motion. A video of this same process for the SO101 as reference can be found [here](https://huggingface.co/docs/lerobot/en/so101#calibration-video)

#### Leader

Do the same steps to calibrate the leader arm, run the following command or API example:

<hfoptions id="calibrate_leader">
<hfoption id="Command">

```bash
lerobot-calibrate \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0 \ # <- The port of your robot
    --teleop.id=my_awesome_leader_arm # <- Give the robot a unique name
```

</hfoption>
<hfoption id="API example">

<!-- prettier-ignore-start -->
```python
from lerobot.teleoperators.so100_leader import SO100LeaderConfig, SO100Leader

config = SO100LeaderConfig(
    port="/dev/tty.usbmodem58760431551",
    id="my_awesome_leader_arm",
)

leader = SO100Leader(config)
leader.connect(calibrate=False)
leader.calibrate()
leader.disconnect()
```
<!-- prettier-ignore-end -->

</hfoption>
</hfoptions>

Congrats 🎉, your robot is all set to learn a task on its own. Start training it by following this tutorial: [Getting started with real-world robots](./getting_started_real_world_robot)

> [!TIP]
> If you have any questions or need help, please reach out on [Discord](https://discord.com/invite/s3KuuzsPFb).