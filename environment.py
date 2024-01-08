import math
import numpy as np
from game import Unicycle, Loop, draw
import constants

# Mediates and abstracts Unicycle/World as MDP.
# Can generate episodes, visualize to pygame


def absolute_angle(x, y):
    angle = math.atan2(y, x)
    if angle < 0:
        angle = math.pi + abs(angle + math.pi)
    return angle


class Environment:
    def __init__(self, canvas=None):
        self.episode_over = True
        # self._action_set = np.array(np.meshgrid(
        #     [-1, 0, 1], [-2, -1, 0, 1, 2])).T.reshape(15, 2)
        self._action_set = np.array([[-1, 1], [0, 1], [1, 1]])
        self.canvas = canvas

    def reset(self):
        self.track = Loop(constants.inner_radius, constants.outer_radius)
        self.unicycle = Unicycle(self.track.generate_starting_position())
        self.episode_over = False
        self.entities = [self.track, self.unicycle]

    def act(self, action):
        # Execute timestep

        if self.episode_over:
            # Throw error
            raise Exception('Tried to act when episode was over')

        x1, y1 = self.unicycle.position
        self.unicycle.step(action, constants.delta_time)
        x2, y2 = self.unicycle.position
        angular_progress = absolute_angle(x2, y2) - absolute_angle(x1, y1)
        # TODO: Handle edge case of starting before then ending after finish line (0 degrees)

        if angular_progress > 6:
            angular_progress = 0

        reward = angular_progress

        # Update failed or not, used to terminate episode
        if not self.episode_over:
            self.episode_over = not self.track.on_track(self.unicycle.position) \
                or abs(self.unicycle.unicycle_roll_angle) > math.pi/2 - 1e-1 \
                or absolute_angle(x2, y2) > math.pi - 0.1

        if self.canvas is not None:
            draw(self.canvas, entities=[self.track, self.unicycle])

        return reward

    # TODO: Better convention?
    def terminated(self):
        return self.episode_over

    def observe(self):
        # Featurize position, momentum, etc. + bias.
        # Could maybe improve featurization? Or try out convnet/visual agent

        # Featurization is also a setting of the environment in this case,
        # not up to the agent.

        x = self.unicycle.position[0]
        y = self.unicycle.position[1]
        heading_angle = self.unicycle.heading_angle
        rider_roll_angle = self.unicycle.rider_lean_angle
        unicycle_roll_angle = self.unicycle.unicycle_roll_angle
        wheel_angular_momentum = self.unicycle.wheel_angular_momentum
        unicycle_roll_momentum = self.unicycle.unicycle_roll_momentum

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

    def n_obs():
        return 10

    def action_set(self):
        return self._action_set
