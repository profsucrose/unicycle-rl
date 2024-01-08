import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()

        # Basic MLP w/ ReLU
        self.stack = torch.nn.Sequential(
            nn.Linear(n_inputs, n_inputs * 2),
            nn.ReLU(),
            nn.Linear(n_inputs * 2, n_inputs * 2),
            nn.ReLU(),
            nn.Linear(n_inputs * 2, n_outputs),
        )

    def forward(self, observation):
        return self.stack.forward(observation)

    def save(self, path, step, optimizer):
        torch.save({
            'step': step,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict()
        }, path)

    def load(self, checkpoint_path, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        step = checkpoint['step']
        self.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
