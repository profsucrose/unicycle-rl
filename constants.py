import math

# TODO: Clean up

# DQN
C = 5
lr = 1e-4
discount_factor = 0.99
replay_buffer_size = 50000
total_episodes = 100000
initial_epsilon = 0.1
min_epsilon = 0.0001
epsilon_discount_rate = 1e-7
batch_size = 32
maximum_checkpoints = 5
save_logs_frequency = 1000
initial_observation_episodes = 100
show_loss_frequency = 100

# For environment, taken from game
inner_radius = 5
outer_radius = 15

rider_height = 4
rider_offset_y = 0.6

wheel_radius = 1.1
wheel_mass = 5
wheel_inertia = 70
gravity = 9.81
friction = 0.9

pedaling_acceleration = 3  # radians
lean_speed = math.radians(200)

fps = 60
delta_time = 1/fps

canvas_width = 400
canvas_height = 300

world_width = 40
world_height = 30

scaling_factor = canvas_width / \
    world_width if world_width < world_height else canvas_height/world_height
