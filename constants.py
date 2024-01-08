import math

# TODO: Clean up


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
