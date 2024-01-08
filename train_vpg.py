import numpy as np
from environment import Environment
from vpg import Agent
import os
import matplotlib.pyplot as plt

os.makedirs('checkpoints', exist_ok=True)

env = Environment()
action_set = env.action_set()
n_obs = env.n_obs()
n_actions = len(action_set)
agent = Agent(n_obs, n_actions)

# Parameters
n_actions = len(action_set)
n_episodes = 1_000_0000
batch_size = 10
discount_factor = 0.99

# Train loop
total_rewards, batch_rewards, batch_observations, batch_actions = [], [], [], []
batch_counter = 1

plot_xaxis = []
plot_yaxis = []

for current_episode in range(n_episodes):
    env.reset()
    observation = env.observe()
    rewards, actions, observations = [], [], []

    while not env.terminated():
        action_probs = agent.predict(observation).detach().numpy()
        action_index = np.random.choice(
            np.arange(len(action_probs)), p=action_probs)
        action = action_set[action_index]

        # Record timestep
        observations.append(observation)

        reward = env.step(action)
        observation = env.observe()

        rewards.append(reward)
        # Use index during training step, not actual action
        actions.append(action_index)

    # Discount rewards (take vector of discount factor, exponentiate, multiply w/ rewards)
    r = np.full(
        len(rewards), discount_factor) ** np.arange(len(rewards)) * np.array(rewards)

    # Discounted rewards are cumulative sum of rewards (reverse, cumsum, reverse) minus baseline (average)
    r = r[::-1].cumsum()[::-1]
    discounted_rewards = r - r.mean()

    # Collect the per-batch rewards, observations, actions
    batch_rewards.extend(discounted_rewards)
    batch_observations.extend(observations)
    batch_actions.extend(actions)

    batch_counter += 1
    total_rewards.append(sum(rewards))

    if batch_counter >= batch_size:
        agent.update(np.array(batch_observations), np.array(
            batch_actions), np.array(batch_rewards))

        # Reset the batch
        batch_rewards, batch_observations, batch_actions = [], [], []
        batch_counter = 1

    # Get running average of last 100 rewards, print every 100 episodes
    if current_episode % 100 == 0:
        average_reward = np.mean(total_rewards[-100:])
        plot_xaxis.append(current_episode)
        plot_yaxis.append(average_reward)
        print(
            f"Average of last 100 rewards as of episode {current_episode}: {average_reward:.2f}")

agent.save(f'checkpoints/checkpoint-{current_episode+1}.pth')

plt.plot(plot_xaxis, plot_yaxis)
plt.xlabel('Episode')
plt.ylabel('Average total reward over last 100 episodes')
plt.show()
