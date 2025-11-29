# SAC Actor vs ACFQL Actor 对比

## 概述

你的代码库中有**两个不同的 Actor 实现**：

1. **SAC Actor** (`lerobot/src/lerobot/rl/actor.py`) - 用于 SAC 策略
2. **ACFQL Actor** (`lerobot/src/lerobot/rl/acfql/actor.py`) - 用于 ACFQL 策略

你提供的代码片段是 **ACFQL Actor** 版本。

## 主要区别

### 1. 策略类型 (Policy Type)

| 特性 | SAC Actor | ACFQL Actor |
|------|-----------|-------------|
| **策略类** | `SACPolicy` | `ACFQLPolicy` |
| **导入路径** | `from lerobot.policies.sac.modeling_sac import SACPolicy` | `from lerobot.policies.acfql.modeling_acfql import ACFQLPolicy` |
| **算法** | Soft Actor-Critic | Actor-Critic Fitted Q-Learning |

### 2. 配置类型 (Config Type)

| 特性 | SAC Actor | ACFQL Actor |
|------|-----------|-------------|
| **配置类** | `TrainRLServerPipelineConfig` | `ACFQLTrainRLServerPipelineConfig` |
| **导入路径** | `from lerobot.configs.train import TrainRLServerPipelineConfig` | `from .configs import ACFQLTrainRLServerPipelineConfig` |

### 3. Processors 处理

#### SAC Actor（简单方式）
```python
# 直接使用 env_processor 和 action_processor
# 没有单独的 preprocessor/postprocessor
action = policy.select_action(batch=observation)
```

#### ACFQL Actor（复杂方式）
```python
# 创建独立的 preprocessor 和 postprocessor
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg.policy,
    pretrained_path=cfg.policy.pretrained_path,
    **processor_kwargs,
    **postprocessor_kwargs,
)

# 对 observation 进行预处理（包括维度转换）
observation_for_inference = preprocessor({
    **{"observation.state": observation["observation.state"]},
    **{k: v.permute(0, 2, 3, 1) for k, v in observation.items() if "observation.images" in k},
})

# 过滤并转换回原始格式
observation_for_inference = {
    **{"observation.state": observation_for_inference["observation.state"]},
    **{k: v.permute(0, 3, 1, 2) for k, v in observation_for_inference.items() if "observation.images" in k},
}

# 使用预处理后的 observation
action = policy.select_action(batch=observation_for_inference)

# 对 action 进行后处理
action = postprocessor(action)
```

**关键区别：**
- **SAC Actor：** 直接使用 observation，无需预处理
- **ACFQL Actor：** 需要预处理（归一化、维度转换）和后处理（反归一化）

### 4. 策略参数更新

#### SAC Actor
```python
# 更新 actor 参数
policy.actor.load_state_dict(actor_state_dict)

# 更新 discrete_critic（如果存在）
if hasattr(policy, "discrete_critic") and "discrete_critic" in state_dicts:
    policy.discrete_critic.load_state_dict(discrete_critic_state_dict)
```

#### ACFQL Actor
```python
# 更新 actor_onestep_flow 参数
policy.actor_onestep_flow.load_state_dict(actor_state_dict, strict=True)

# 注释掉了 discrete_critic 的更新
# if hasattr(policy, "discrete_critic") and "discrete_critic" in state_dicts:
#     ...
```

**关键区别：**
- **SAC Actor：** 更新 `policy.actor`
- **ACFQL Actor：** 更新 `policy.actor_onestep_flow`

### 5. 初始参数等待机制

#### SAC Actor
```python
# 等待初始参数，但有超时机制（5秒）
max_wait_time = 5.0
start_time = time.time()
parameters_received = False
while True:
    try:
        bytes_state_dict = parameters_queue.get(timeout=0.5)
        # ... 加载参数 ...
        parameters_received = True
        break
    except Empty:
        elapsed = time.time() - start_time
        if elapsed > max_wait_time:
            logging.warning("Timeout waiting for initial parameters")
            break
```

#### ACFQL Actor
```python
# 如果 offline_steps > 0，则等待初始参数（阻塞等待）
if cfg.policy.offline_steps > 0:
    logging.info("[ACTOR] Waiting for initial policy parameters from learner")
    update_policy_parameters(
        policy=policy, 
        parameters_queue=parameters_queue, 
        device=device, 
        wait_for_update=True  # 阻塞等待
    )
```

**关键区别：**
- **SAC Actor：** 有超时机制，超时后继续执行
- **ACFQL Actor：** 如果 `offline_steps > 0`，会阻塞等待，直到收到参数

### 6. Intervention 处理

#### SAC Actor
```python
if intervention_info.get(TeleopEvents.IS_INTERVENTION, False):
    episode_intervention = True
    episode_intervention_steps += 1
    # 没有重置策略状态
```

#### ACFQL Actor
```python
if intervention_info.get(TeleopEvents.IS_INTERVENTION, False):
    episode_intervention = True
    episode_intervention_steps += 1
    policy.reset()  # 重置策略状态（用于清除 action queue）
```

**关键区别：**
- **SAC Actor：** 不重置策略状态
- **ACFQL Actor：** 在干预时重置策略状态，并在 episode 结束时也重置

### 7. Episode 结束处理顺序

#### SAC Actor
```python
if done or truncated:
    logging.info(f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}")
    
    update_policy_parameters(...)  # 先更新参数
    
    if len(list_transition_to_send_to_learner) > 0:
        push_transitions_to_transport_queue(...)  # 再发送 transitions
    
    # ... 发送 interactions ...
```

#### ACFQL Actor
```python
if done or truncated:
    episode_time = time.perf_counter() - episode_start_time
    logging.info(f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}, Episode time: {episode_time:.2f}s")
    
    # 先发送 transitions
    if len(list_transition_to_send_to_learner) > 0:
        push_transitions_to_transport_queue(...)
    
    # 再发送 interactions
    interactions_queue.put(...)
    
    # 最后更新参数
    update_policy_parameters(...)
    
    policy.reset()  # 重置策略状态
```

**关键区别：**
- **SAC Actor：** 先更新参数，再发送数据
- **ACFQL Actor：** 先发送数据，再更新参数，并且重置策略状态

### 8. FPS 跟踪

#### SAC Actor
```python
policy_timer = TimerManager("Policy inference", log=False)
# 只有 policy 推理的计时
```

#### ACFQL Actor
```python
policy_timer = TimerManager("Policy inference", log=False)
fps_tracker = TimerManager("Episode FPS", log=False)  # 额外的 episode FPS 跟踪
episode_started = True
episode_start_time = time.perf_counter()

# 跟踪每个 episode 的时间
```

**关键区别：**
- **SAC Actor：** 只跟踪 policy 推理的 FPS
- **ACFQL Actor：** 同时跟踪 policy FPS 和 episode FPS

### 9. 日志记录

#### SAC Actor
```python
# 基本的日志记录
logging.info("make_policy")
logging.info(f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}")
```

#### ACFQL Actor
```python
# 更详细的日志记录
logging.info(pformat(cfg.to_dict()))  # 打印完整配置
logging.info(f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}, Episode time: {episode_time:.2f}s")
logging.info(", ".join([f"{k} : {v:.2f}" for k, v in stats.items()]))  # 打印统计信息
```

### 10. 其他差异

| 特性 | SAC Actor | ACFQL Actor |
|------|-----------|-------------|
| **调试日志** | 有 `[ACTOR DEBUG]` 日志 | 无 |
| **Config 打印** | 无 | 有 `pformat(cfg.to_dict())` |
| **Episode 时间记录** | 无 | 有 |
| **`register_third_party_devices()`** | 无 | 有（在 `__main__` 中） |

## 功能对比总结

| 功能 | SAC Actor | ACFQL Actor |
|------|-----------|-------------|
| **算法** | SAC | ACFQL |
| **Processors** | 简单（直接在 env/action processor 中） | 复杂（独立的 pre/post processors） |
| **初始参数等待** | 有超时机制 | 阻塞等待（如果 offline_steps > 0） |
| **Intervention 处理** | 不重置策略 | 重置策略状态 |
| **Episode 重置** | 不重置策略 | 重置策略状态 |
| **FPS 跟踪** | 仅 policy FPS | Policy FPS + Episode FPS |
| **日志详细程度** | 基本 | 更详细 |

## 使用场景

### SAC Actor 适用于：
- 标准的 SAC 策略训练
- 简单的预处理需求
- 不需要策略状态重置的场景

### ACFQL Actor 适用于：
- ACFQL 策略训练
- 需要复杂预处理（归一化、维度转换）的场景
- 需要策略状态重置的场景（chunking 支持）
- 需要详细日志和统计的场景

## 建议

如果你当前使用的是 **SAC 策略**（`configs/simulation/train_gym_hil_env.json` 中 `policy.type: "sac"`），应该使用：

**SAC Actor** (`lerobot/src/lerobot/rl/actor.py`)

如果你需要使用 **ACFQL 策略**，应该使用：

**ACFQL Actor** (`lerobot/src/lerobot/rl/acfql/actor.py`)

**注意：** Actor 和 Learner 的版本必须匹配（都使用 SAC 或都使用 ACFQL）。

