import numpy as np
import pygame
from environment import Environment
import constants
from vpg import Agent
import glob

pygame.init()

canvas = pygame.display.set_mode(
    (constants.canvas_width, constants.canvas_height))
clock = pygame.time.Clock()

pygame.display.set_caption('Agent Demo')

env = Environment(canvas=canvas)

action_set = env.action_set()
n_obs = env.n_obs()
n_actions = len(action_set)
agent = Agent(n_obs, n_actions)

checkpoints = glob.glob('checkpoints/*.pth')

if checkpoints:
    latest = max(int(s[s.index('-')+1:-4]) for s in checkpoints)
    checkpoint = f'checkpoints/checkpoint-{latest}.pth'
    agent.load(checkpoint)
    print(f'Found checkpoint! Loaded {checkpoint}')

env.reset()

running = True

total_reward = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not env.terminated():
        state = env.observe()

        action_probs = agent.predict(state).detach().numpy()
        action_index = np.random.choice(
            np.arange(len(action_probs)), p=action_probs)
        action = action_set[action_index]

        reward = env.step(action)
        total_reward += reward

    print('return', total_reward)
    pygame.display.flip()
    clock.tick(constants.fps)

pygame.quit()
