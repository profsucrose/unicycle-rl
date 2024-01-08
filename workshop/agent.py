import numpy as np
import math


class Agent:
    def featurize_state(state):
        x = state['x']
        y = state['y']
        heading_angle = state['heading_angle']
        rider_roll_angle = state['rider_roll_angle']
        unicycle_roll_angle = state['unicycle_roll_angle']
        wheel_angular_momentum = state['wheel_angular_momentum']
        unicycle_roll_momentum = state['unicycle_roll_momentum']

        r = math.hypot(x, y)
        cos_theta, sin_theta = x/r, y/r
        cos_heading = math.cos(heading_angle)
        sin_heading = math.sin(heading_angle)
        rider_roll = rider_roll_angle/(math.pi/2)
        lean = unicycle_roll_angle/(math.pi/2)
        wheel_momentum = wheel_angular_momentum/6  # Max asymptotic wheel omega

        # Angular vel, so no real meaning for unit
        lean_momentum = unicycle_roll_momentum

        # Flip sign of lean to be consistent? (I didn't bother figuring this out)
        lean *= -1
        lean_momentum *= -1

        # Add 1 as a last entry as a bias term an agent can incorporate
        return np.array([r, cos_theta, sin_theta, cos_heading, sin_heading, rider_roll, lean, wheel_momentum, lean_momentum, 1])
