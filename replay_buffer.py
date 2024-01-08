import numpy as np
import random
import constants
from collections import deque


class ReplayBuffer():
    def __init__(self):
        self.buffer = deque()  # Deque as ring buffer

    def record(self, state, action, reward, state_new, terminal):
        if len(self.buffer) > constants.replay_buffer_size:
            self.buffer.popleft()
        self.buffer.append((state, action, reward, state_new, terminal))

    def sample(self, batch_size):
        # TODO: Can probably do this directly with numpy?

        batch = random.sample(self.buffer, batch_size)

        batch_state = []
        batch_action = []
        batch_reward = []
        batch_state_new = []
        batch_terminal = []

        for state, action, reward, state_new, terminal in batch:
            batch_state.append(state)
            batch_action.append(action)
            batch_reward.append(reward)
            batch_state_new.append(state_new)
            batch_terminal.append(terminal)

        batch_state = np.stack(batch_state)
        batch_action = np.stack(batch_action)
        batch_reward = np.stack(batch_reward)
        batch_state_new = np.stack(batch_state_new)
        batch_terminal = np.stack(batch_terminal)

        return batch_state, batch_action, batch_reward, batch_state_new, batch_terminal
