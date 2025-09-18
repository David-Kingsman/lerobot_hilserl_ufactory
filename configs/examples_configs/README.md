---
license: apache-2.0
---

A repository that contains example json files that can be used for different applications of the [LeRobot](https://github.com/huggingface/lerobot) code base.

Current available configs:
 - `env_config_so100.json`: real robot environment configuration to be used to teleoperate, record dataset and replay a dataset on the real robot with the `lerobot/scripts/rl/gym_manipulator.py` script.
 - `train_config_hilserl_so100.json`: training config on the real robot with the HILSerl RL framework in LeRobot.
 - `gym_hil_env.json`: simulated environment configuration for the [gym_hil](https://github.com/huggingface/gym-hil/) mujoco based env. 
 - `train_gym_hil_env.json`: training configuration for gym_hil with the HILSerl RL framework in LeRobot. 
 - `reward_classifier_train_config.json`: configuration to train a reward classifier with LeRobot. 