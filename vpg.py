import torch
from torch import nn, optim


class Agent:
    def __init__(self, n_obs, n_actions):
        self.n_obs = n_obs
        self.n_actions = n_actions

        self.network = nn.Sequential(
            nn.Linear(self.n_obs, 20),
            nn.ReLU(),
            nn.Linear(20, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.network.parameters(), lr=0.01)

    def predict(self, obs):
        return self.network(torch.FloatTensor(obs))

    def load(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def update(self, batch_observations, batch_actions, batch_rewards):
        self.optimizer.zero_grad()

        batch_observations = torch.FloatTensor(batch_observations)
        batch_actions = torch.LongTensor(batch_actions)
        batch_rewards = torch.FloatTensor(batch_rewards)

        logprob = torch.log(self.predict(batch_observations))
        batch_actions = batch_actions.reshape(
            len(batch_actions), 1)
        selected_logprobs = batch_rewards * \
            torch.gather(logprob, 1, batch_actions).squeeze()
        loss = -selected_logprobs.mean()

        # Backprop/optimize
        loss.backward()
        self.optimizer.step()
