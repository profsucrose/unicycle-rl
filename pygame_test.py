
import pygame
from environment import Environment
import constants

pygame.init()

canvas = pygame.display.set_mode(
    (constants.canvas_width, constants.canvas_height))
clock = pygame.time.Clock()

pygame.display.set_caption('Agent Demo')

env = Environment(canvas=canvas)
action_set = env.action_set()

env.reset()

running = True

key_held = {
    pygame.K_w: False,
    pygame.K_a: False,
    pygame.K_s: False,
    pygame.K_d: False,
    pygame.K_LSHIFT: False,
}

total_reward = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key in key_held:
                key_held[event.key] = True
        if event.type == pygame.KEYUP:
            if event.key in key_held:
                key_held[event.key] = False

    leaning = int(key_held[pygame.K_d]) - int(key_held[pygame.K_a])
    pedaling = int(key_held[pygame.K_w]) - int(key_held[pygame.K_s])
    if key_held[pygame.K_LSHIFT]:
        pedaling *= 2

    action = [leaning, pedaling]
    if not env.terminated():
        reward = env.act(action)
        total_reward += reward
    print('return', total_reward)
    pygame.display.flip()
    clock.tick(constants.fps)

pygame.quit()
