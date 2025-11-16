# BC预训练工作流程（用于HiLSERL）

本指南说明如何使用同一个数据集先进行BC（Behavior Cloning）预训练，然后再进行HiLSERL训练，避免从零开始学习RL。

## 工作流程概述

这是一个更高效的工作流程，**类似于reward classifier的使用方式**：

1. **收集演示数据**（用于BC预训练和reward classifier）
2. **独立训练BC模型**（如ACT）- 使用收集的数据集，这是一个**独立的训练步骤**
3. **独立训练Reward Classifier**（可选）- 使用相同的数据集，这也是**独立的训练步骤**
4. **HiLSERL训练** - 在配置中指定BC预训练路径和reward classifier路径，learner会自动加载它们作为先验

**重要说明**：
- BC预训练和HiLSERL训练是**分开的、顺序执行的**，不是同时运行
- BC预训练完成后，将模型路径配置到HiLSERL训练配置中
- HiLSERL learner启动时会自动加载BC预训练的编码器权重来初始化SAC策略
- 这完全类似于reward classifier的工作方式：先训练，再在配置中指定路径使用

## 完整步骤

### 步骤1: 收集演示数据

使用 `gym_manipulator.py` 收集演示数据。这个数据集将同时用于：
- BC预训练（ACT）
- Reward Classifier训练

示例配置：`env_config_hilserl_xarm6_gamepad.json`

```json
{
  "env": {
    // ... 环境配置 ...
  },
  "dataset": {
    "repo_id": "your_username/dataset_name",
    "root": "/path/to/dataset",
    "task": "pick_cube",
    "num_episodes_to_record": 20,
    "push_to_hub": false
  },
  "mode": "record",
  "device": "cpu"
}
```

收集数据：
```bash
python -m lerobot.scripts.rl.gym_manipulator --config_path configs/your_env_config.json
```

### 步骤2: 训练BC模型（ACT）

使用步骤1收集的数据集训练ACT模型。

创建ACT训练配置文件：`train_config_act_bc.json`

```json
{
  "dataset": {
    "repo_id": "your_username/dataset_name",
    "root": "/path/to/dataset"
  },
  "policy": {
    "type": "act",
    "n_obs_steps": 1,
    "chunk_size": 100,
    "n_action_steps": 100,
    "vision_backbone": "resnet18",
    "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
    "input_features": {
      "observation.images.webcam_1": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      },
      "observation.images.realsense": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      },
      "observation.state": {
        "type": "STATE",
        "shape": [15]
      }
    },
    "output_features": {
      "action": {
        "type": "ACTION",
        "shape": [6]
      }
    },
    "device": "cuda",
    "use_amp": true
  },
  "batch_size": 8,
  "num_workers": 4,
  "steps": 50000,
  "eval_freq": 5000,
  "save_freq": 10000,
  "save_checkpoint": true,
  "output_dir": null,
  "job_name": "act_bc_pretrain",
  "wandb": {
    "enable": true,
    "project": "bc_pretrain"
  }
}
```

训练ACT：
```bash
lerobot-train --config_path train_config_act_bc.json
```

训练完成后，ACT模型将保存在 `outputs/train/YYYY-MM-DD/HH-MM-SS_act_bc_pretrain/checkpoints/last/pretrained_model/`

### 步骤3: 训练Reward Classifier（可选）

使用相同的数据集训练reward classifier。

```bash
lerobot-train --config_path configs/reward_classifier_train_config.json
```

### 步骤4: HiLSERL训练（使用BC预训练作为先验）

**重要**：BC预训练和HiLSERL训练是**分开执行的**。先完成步骤2的BC训练，然后在HiLSERL配置中指定预训练路径。

在HiLSERL训练配置的 `policy` 部分添加 `bc_pretrain` 配置，指向步骤2训练的ACT模型：

修改 `train_config_hilserl_xarm6.json`：

```json
{
  "policy": {
    "type": "sac",
    "vision_encoder_name": "helper2424/resnet10",
    "bc_pretrain": {
      "pretrained_path": "/path/to/outputs/train/YYYY-MM-DD/HH-MM-SS_act_bc_pretrain/checkpoints/last/pretrained_model",
      "policy_type": "act"
    },
    "freeze_vision_encoder": true,
    // ... 其他SAC配置 ...
  },
  "env": {
    "processor": {
      "reward_classifier": {
        "pretrained_path": "/path/to/reward_classifier/checkpoints/last/pretrained_model",
        "success_threshold": 0.5,
        "success_reward": 1.0
      }
    }
  },
  // ... 其他配置 ...
}
```

**配置说明**：
- `bc_pretrain.pretrained_path`: BC预训练模型的路径（可以是本地路径或HuggingFace repo ID）
- `bc_pretrain.policy_type`: BC模型类型，目前支持 `"act"`（默认值）

**工作流程**：
1. HiLSERL learner启动时，会检查 `policy.bc_pretrain.pretrained_path` 配置
2. 如果指定了路径，learner会**自动加载**指定类型的BC模型（如ACT）的编码器权重
3. 使用这些权重初始化SAC策略的视觉编码器
4. 然后开始正常的HiLSERL训练流程

启动learner（会自动加载BC预训练权重）：
```bash
python -m lerobot.scripts.rl.learner --config_path train_config_hilserl_xarm6.json
```

启动actor：
```bash
python -m lerobot.scripts.rl.actor --config_path train_config_hilserl_xarm6.json
```

## 优势

使用BC预训练的HiLSERL工作流程有以下优势：

1. **更快的收敛**：RL从预训练的编码器开始，而不是从零开始学习视觉特征
2. **更好的样本效率**：减少需要的人类干预和在线交互
3. **更稳定的训练**：预训练的编码器提供了更好的特征表示
4. **数据复用**：同一个数据集可以用于BC预训练和reward classifier训练

## 注意事项

1. **编码器兼容性**：确保ACT和SAC使用兼容的视觉编码器（如ResNet）
2. **数据集对齐**：确保BC训练和HiLSERL训练使用的观察空间一致
3. **冻结编码器**：在HiLSERL训练中，建议设置 `freeze_vision_encoder: true` 来保持预训练特征
4. **权重加载**：如果架构不完全匹配，系统会尝试加载匹配的权重，不匹配的部分使用随机初始化

## 完整工作流程示例

```bash
# ============================================
# 阶段1: 数据收集和预训练（独立执行）
# ============================================

# 1. 收集演示数据（用于BC和reward classifier）
python -m lerobot.scripts.rl.gym_manipulator \
  --config_path configs/env_config_hilserl_xarm6_gamepad.json

# 2. 训练BC模型（ACT）- 独立训练步骤
lerobot-train --config_path configs/train_config_act_bc.json
# 训练完成后，记录模型路径，例如：
# outputs/train/2025-01-15/10-30-00_act_bc_pretrain/checkpoints/last/pretrained_model

# 3. 训练Reward Classifier（可选）- 独立训练步骤
lerobot-train --config_path configs/reward_classifier_train_config.json
# 训练完成后，记录模型路径

# ============================================
# 阶段2: HiLSERL训练（使用预训练模型作为先验）
# ============================================

# 4. 在 train_config_hilserl_xarm6.json 中配置预训练路径：
#    - policy.bc_pretrain.pretrained_path: 指向步骤2的ACT模型
#    - policy.bc_pretrain.policy_type: 设置为 "act"
#    - env.processor.reward_classifier.pretrained_path: 指向步骤3的reward classifier

# 5. 启动HiLSERL训练（learner会自动加载BC预训练权重）
python -m lerobot.scripts.rl.learner \
  --config_path configs/train_config_hilserl_xarm6.json

# 6. 在另一个终端启动actor
python -m lerobot.scripts.rl.actor \
  --config_path configs/train_config_hilserl_xarm6.json
```

## 关键理解

**BC预训练和HiLSERL的关系**：
- ✅ **正确理解**：BC预训练是独立的训练步骤，完成后在HiLSERL配置中指定路径，learner启动时自动加载
- ❌ **错误理解**：BC和HiLSERL同时运行，或者需要手动加载权重

**类似reward classifier**：
- Reward classifier：先训练 → 在配置中指定路径 → 自动使用
- BC预训练：先训练 → 在配置中指定路径 → 自动使用

通过这个工作流程，你的HiLSERL训练将从一个已经学习到基本视觉特征的编码器开始，而不是完全从零开始，这可以显著提高训练效率和最终性能。

