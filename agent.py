import math
import numpy as np
import random
from model import Model
import constants
from torch import optim
from torch.nn.functional import mse_loss
import torch
import os
import glob


class Agent:
    def __init__(self, action_set):
        n_state_features = 10
        self.Q_network = Model(n_state_features, len(action_set))
        self.target_network = Model(n_state_features, len(action_set))
        self.optimizer = optim.Adam(
            self.Q_network.parameters(), lr=constants.lr)
        self.epsilon = constants.initial_epsilon
        self.action_set = action_set
        self.n_actions = len(action_set)

    def take_action(self, state):
        # Agent expects state to be featurized
        # TODO: Clean this up
        featurized = torch.from_numpy(state).float()

        self.Q_network.eval()
        max_action = self.Q_network.forward(featurized).max(dim=0)

        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions-1)
        else:
            return max_action.indices.item()

    def update_target_network(self):
        # copy current_network to target network
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def update_Q_network(self, state, action, reward, state_new, terminal):
        state = torch.from_numpy(state).float()
        action = torch.from_numpy(action).float()
        reward = torch.from_numpy(reward).float()
        state_new = torch.from_numpy(state_new).float()
        terminal = torch.from_numpy(terminal).float()

        self.Q_network.eval()
        self.target_network.eval()

        action_new = self.Q_network.forward(
            state_new).max(dim=1).indices.view(-1, 1)
        action_new_onehot = torch.zeros(constants.batch_size, self.n_actions)
        action_new_onehot = action_new_onehot.scatter_(1, action_new, 1.0)

        # Calculate one-step backup: y = r + discount_factor * Q_target(s', a')
        y = (reward + torch.mul(((self.target_network.forward(state_new)
             * action_new_onehot).sum(dim=1) * terminal), constants.discount_factor))

        # Regress Q(s, a) -> y
        self.Q_network.train()
        Q = (self.Q_network.forward(state) * action).sum(dim=1)
        loss = mse_loss(input=Q, target=y.detach())

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_epsilon(self):
        if self.epsilon > constants.min_epsilon:
            self.epsilon -= constants.epsilon_discount_rate

    def stop_epsilon(self):
        self.epsilon_tmp = self.epsilon
        self.epsilon = 0

    def restore_epsilon(self):
        self.epsilon = self.epsilon_tmp

    def save(self, step, checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
        model_list = glob.glob(os.path.join(checkpoints_path, '*.pth'))
        if len(model_list) > constants.maximum_checkpoints - 1:
            min_step = min([int(li.split('/')[-1][6:-4]) for li in model_list])
            os.remove(os.path.join(checkpoints_path,
                      'model-{}.pth' .format(min_step)))
        checkpoints_path = os.path.join(
            checkpoints_path, 'model-{}.pth' .format(step))
        self.Q_network.save(checkpoints_path, step=step,
                            optimizer=self.optimizer)
        print('Saved checkpoint {}' .format(checkpoints_path))

    def try_restore_latest(self, checkpoints_path):
        checkpoints = glob.glob(os.path.join('checkpoints', 'dqn', '*.pth'))
        if not checkpoints:
            return
        latest = max(int(li.split('/')[1][6:-4]) for li in checkpoints)
        path = os.path.join(checkpoints_path, f'model-{latest}.pth')
        print('Found checkpoint! Restoring...')
        self.restore(path)

    def restore(self, checkpoint_path):
        self.Q_network.load(checkpoint_path)
        self.target_network.load(checkpoint_path)
        print('Restored from checkpoint {}' .format(checkpoint_path))
