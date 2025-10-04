import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# check new dataset
data_df = pd.read_parquet('/home/zekaijin/lerobot-hilserl/datasets/lite6_push_cube_reward_classifier_test2/data/chunk-000/file-000.parquet')

print('=== data analysis ===')
print(f'total frames: {len(data_df)}')
episode_col = "episode_index"
print(f'total episodes: {data_df[episode_col].nunique()}')
print()

# analyze reward distribution
rewards = data_df['next.reward']
print('reward distribution:')
print(f'  0 reward frames: {(rewards == 0).sum()} ({(rewards == 0).mean()*100:.1f}%)')
print(f'  1 reward frames: {(rewards == 1).sum()} ({(rewards == 1).mean()*100:.1f}%)')
print()

# analyze reward distribution for each episode
print('reward distribution for each episode:')
success_episodes = 0
total_reward_frames = 0
for ep in sorted(data_df[episode_col].unique()):
    ep_data = data_df[data_df[episode_col] == ep]
    reward_frames = (ep_data['next.reward'] > 0).sum()
    total_frames = len(ep_data)
    if reward_frames > 0:
        success_episodes += 1
        total_reward_frames += reward_frames

print(f'success episodes: {success_episodes}/{data_df[episode_col].nunique()}')
print(f'success rate: {success_episodes/data_df[episode_col].nunique()*100:.1f}%')
print(f'total reward frames: {total_reward_frames}')
print(f'average reward frames per success episode: {total_reward_frames/success_episodes:.1f}' if success_episodes > 0 else 'no success episodes')
print(f'data imbalance ratio: {(rewards == 0).sum() / (rewards == 1).sum():.1f}:1' if (rewards == 1).sum() > 0 else 'no positive samples')

print()
print('=== visualization analysis ===')

# set font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Reward Distribution Analysis', fontsize=16, fontweight='bold')

# 1. total reward distribution pie chart
reward_counts = [rewards.sum(), len(rewards) - rewards.sum()]
labels = ['Reward=1', 'Reward=0']
colors = ['#ff9999', '#66b3ff']
ax1.pie(reward_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title('Overall Reward Distribution')

# 2. reward frames per episode bar chart
episode_rewards = []
episode_names = []
for ep in sorted(data_df[episode_col].unique()):
    ep_data = data_df[data_df[episode_col] == ep]
    reward_frames = (ep_data['next.reward'] > 0).sum()
    total_frames = len(ep_data)
    episode_rewards.append(reward_frames)
    episode_names.append(f'Episode {ep}\n({reward_frames}/{total_frames})')

bars = ax2.bar(episode_names, episode_rewards, color='lightcoral', alpha=0.7)
ax2.set_title('Reward Frames per Episode')
ax2.set_ylabel('Number of Reward Frames')
ax2.tick_params(axis='x', rotation=45)

# add value labels to the bar chart
for bar, reward in zip(bars, episode_rewards):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{reward}', ha='center', va='bottom', fontweight='bold')

# 3. reward ratio per episode
episode_ratios = []
for ep in sorted(data_df[episode_col].unique()):
    ep_data = data_df[data_df[episode_col] == ep]
    reward_ratio = (ep_data['next.reward'] > 0).mean() * 100
    episode_ratios.append(reward_ratio)

bars3 = ax3.bar(episode_names, episode_ratios, color='lightgreen', alpha=0.7)
ax3.set_title('Reward Ratio per Episode (%)')
ax3.set_ylabel('Reward Ratio (%)')
ax3.set_ylim(0, 100)
ax3.tick_params(axis='x', rotation=45)

# add percentage labels to the bar chart
for bar, ratio in zip(bars3, episode_ratios):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{ratio:.1f}%', ha='center', va='bottom', fontweight='bold')

# 4. reward timeline
ax4.set_title('Reward Timeline')
colors_timeline = ['red' if r == 1 else 'blue' for r in rewards]
ax4.scatter(range(len(rewards)), [1]*len(rewards), c=colors_timeline, alpha=0.6, s=1)
ax4.set_xlabel('Frame Index')
ax4.set_ylabel('Reward (0/1)')
ax4.set_ylim(0.5, 1.5)

# add episode boundaries
episode_boundaries = []
current_ep = data_df[episode_col].iloc[0]
for i, ep in enumerate(data_df[episode_col]):
    if ep != current_ep:
        episode_boundaries.append(i)
        current_ep = ep

for boundary in episode_boundaries:
    ax4.axvline(x=boundary, color='black', linestyle='--', alpha=0.7, linewidth=1)

plt.tight_layout()
plt.savefig('/home/zekaijin/lerobot-hilserl/test/reward_analysis.png', dpi=300, bbox_inches='tight')
print('figures saved to: /home/zekaijin/lerobot-hilserl/test/reward_analysis.png')

# 显示详细统计
print('\ndetailed statistics:')
for i, ep in enumerate(sorted(data_df[episode_col].unique())):
    ep_data = data_df[data_df[episode_col] == ep]
    reward_frames = (ep_data['next.reward'] > 0).sum()
    total_frames = len(ep_data)
    reward_ratio = reward_frames / total_frames * 100
    
    # 奖励分布统计
    rewards_ep = ep_data['next.reward']
    unique_rewards = sorted(rewards_ep.unique())
    reward_counts = rewards_ep.value_counts().sort_index()
    
    print(f'Episode {ep}:')
    print(f'  total frames: {total_frames}')
    print(f'  reward frames: {reward_frames}')
    print(f'  reward ratio: {reward_ratio:.1f}%')
    print(f'  reward distribution: {unique_rewards[0]} 到 {unique_rewards[-1]} (范围: {unique_rewards})')
    print(f'  reward statistics: {dict(reward_counts)}')
    
    # if there are reward=1 frames, show consecutive reward segments
    if reward_frames > 0:
        reward_positions = rewards_ep[rewards_ep > 0].index.tolist()
        print(f'  reward frame positions: {reward_positions}')
        
        # calculate consecutive reward segments
        consecutive_segments = []
        if reward_positions:
            start = reward_positions[0]
            end = start
            for i in range(1, len(reward_positions)):
                if reward_positions[i] == reward_positions[i-1] + 1:
                    end = reward_positions[i]
                else:
                    consecutive_segments.append((start, end))
                    start = reward_positions[i]
                    end = start
            consecutive_segments.append((start, end))
            
            print(f'  consecutive reward segments: {consecutive_segments}')
            print(f'  number of consecutive reward segments: {len(consecutive_segments)}')
    
    print()

plt.show()