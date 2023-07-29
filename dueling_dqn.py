import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from duelingDQN.dueling_ddqn import DuelingAgent
from baselines_wrappers.wrappers_baselines import make_atari, wrap_deepmind, wrap_pytorch

LOG_DIR = './logs/atari_dueling'

def mini_batch_train_frames(env, agent, max_frames, batch_size):
    episode_rewards = []
    state = env.reset()
    episode_reward = 0

    for frame in range(max_frames):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.append((state, action, reward, next_state, done))
        episode_reward += reward

        if len(agent.replay_buffer) > batch_size:
            agent.update(batch_size)

        if done:
            episode_rewards.append(episode_reward)
            print("Frame " + str(frame) + ": " + str(episode_reward))
            summary_writer.add_scalar('episode_reward', episode_reward, global_step=frame)
            state = env.reset()
            episode_reward = 0

        state = next_state

    return episode_rewards

env_id = "BreakoutNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)

MAX_FRAMES = 1000000
BATCH_SIZE = 32

agent = DuelingAgent(env, use_conv=True)
if torch.cuda.is_available():
    agent.model.cuda()

summary_writer = SummaryWriter(LOG_DIR)
episode_rewards = mini_batch_train_frames(env, agent, MAX_FRAMES, BATCH_SIZE)