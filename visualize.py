
import pygame
from environment import Environment
import constants
from agent import Agent

pygame.init()

canvas = pygame.display.set_mode(
    (constants.canvas_width, constants.canvas_height))
clock = pygame.time.Clock()

pygame.display.set_caption('Agent Demo')

env = Environment(canvas=canvas)
action_set = env.action_set()
agent = Agent(action_set)
agent.try_restore_latest('checkpoints')
# agent.stop_epsilon()  # Act only greedily

env.reset()

running = True

total_reward = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not env.terminated():
        state = env.observe()
        action = agent.take_action(state)
        print(action_set[action])
        reward = env.act(action_set[action])
        total_reward += reward

    print('return', total_reward)
    pygame.display.flip()
    clock.tick(constants.fps)

pygame.quit()
