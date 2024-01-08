import constants
from environment import Environment
from agent import Agent
from replay_buffer import ReplayBuffer
import numpy as np
import os

env = Environment()
action_set = env.action_set()
agent = Agent(action_set)
agent.try_restore_latest('checkpoints')

n_actions = len(action_set)

replay_buffer = ReplayBuffer()

# Training loop
for episode in range(1, constants.total_episodes + 1):
    env.reset()

    state_new = env.observe()

    total_reward = 0
    t_alive = 0

    while not env.terminated():
        state = state_new
        action = agent.take_action(state)
        reward = env.act(action_set[action])
        terminal = env.terminated()
        state_new = env.observe()
        action_onehot = np.zeros(n_actions)
        action_onehot[action] = 1.0
        replay_buffer.record(state, action_onehot, reward, state_new, terminal)

        total_reward += reward

        t_alive += 1

    if episode > constants.initial_observation_episodes:
        batch_state, batch_action, batch_reward, batch_state_new, batch_terminal = replay_buffer.sample(
            constants.batch_size)

        loss = agent.update_Q_network(batch_state, batch_action,
                                      batch_reward, batch_state_new, batch_terminal)

        if episode % constants.C == 0:
            agent.update_target_network()

        if episode % constants.save_logs_frequency == 0:
            agent.save(episode, 'checkpoints')

        agent.update_epsilon()

        if episode % constants.show_loss_frequency == 0:
            print('Episode: {}. t: {}. Reward: {:.3f}. Loss: {:.3f}'.format(
                episode, t_alive, total_reward, loss))
