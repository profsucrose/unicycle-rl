import math
import numpy as np
import math
import random
import constants
import pygame

# Construct classes/helpers for 2d equivalent of the game,
# with numerically equivalent dynamics.


def lerp(a, b, t):
    return t*b + (1-t)*a


def invlerp(a, b, c):
    return (c-a)/(b-a)


def create_camera(w, h):
    return lambda x, y: scale(x+w/2, h-(y+h/2))


def s(n):
    return n * constants.scaling_factor


def scale(w, h):
    return (w * constants.scaling_factor, h * constants.scaling_factor)


def rotate(x, y, rad):
    return (x * math.cos(rad) - y * math.sin(rad), x * math.sin(rad) + y * math.cos(rad))


camera = create_camera(constants.world_width, constants.world_height)


class Loop:
    def __init__(self, inner_radius, outer_radius):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def on_track(self, position):
        r = math.hypot(position[0], position[1])
        return r > self.inner_radius and r < self.outer_radius

    def generate_starting_position(self):
        r = random.randrange(self.inner_radius, self.outer_radius)
        return [r, 0]

    def draw(self, surface: pygame.Surface):
        pygame.draw.circle(surface, (127, 127, 127), camera(
            0, 0), s(self.outer_radius))
        pygame.draw.circle(surface, (255, 255, 255, 255), camera(
            0, 0), s(self.inner_radius))
        pygame.draw.line(surface, (0, 0, 0), camera(
            self.inner_radius, 0), camera(self.outer_radius, 0))


class Unicycle:
    def __init__(self, position=[0, 0]):
        self.position = position
        self.heading_angle = math.pi/2
        self.heading = [math.cos(self.heading_angle),
                        math.sin(self.heading_angle)]

        # Unicycle parameters
        self.pedaling_acceleration = constants.pedaling_acceleration
        self.lean_speed = constants.lean_speed
        self.wheel_radius = constants.wheel_radius
        self.rider_height = constants.rider_height
        self.rider_offset_y = constants.rider_offset_y
        self.wheel_mass = constants.wheel_mass
        self.wheel_inertia = constants.wheel_inertia
        self.gravity = constants.gravity
        self.friction = constants.friction

        # Unicycle state (radians)
        self.rider_lean_angle = 0  # Lean of rider wrt front of unicycle
        self.unicycle_roll_angle = 0  # Lean of unicycle wrt track
        self.unicycle_roll_momentum = 0  # Angular momentum of unicycle
        self.wheel_angular_momentum = 0

    def draw(self, surface: pygame.Surface):
        def transform(x, y):
            # scale, rotate, translate
            # rotate, translate, scale (???)
            p = rotate(x, y, self.heading_angle)
            return camera(p[0] + self.position[0], p[1] + self.position[1])

        length = 2
        width = 1

        # Osculating circle
        if abs(self.unicycle_roll_angle) > 1e-3:
            denom = lerp(0, 1, invlerp(
                0, math.pi/2, abs(self.unicycle_roll_angle)))
            k = math.inf if abs(denom) < 1e-6 else 1/denom
            # TODO: Fix how scaling works w/ camera + scale
            center = transform(0, math.copysign(k, self.unicycle_roll_angle))
            pygame.draw.circle(surface, (255, 255, 0), center, s(k), width=2)

        # "Wheel"/body
        pygame.draw.polygon(surface, (0, 255, 0), (transform(-length/2, width/2), transform(
            length/2, width/2), transform(length/2, -width/2), transform(-length/2, -width/2)))

        # Heading arrow
        pygame.draw.line(surface, (255, 255, 0), transform(
            length/2, 0), transform(length/2 + 2, 0), width=2)

        # Wheel roll helper arrow
        color = (0, 0, 255) if self.unicycle_roll_angle > 0 else (255, 0, 255)
        # pygame.draw.line(surface, color, transform(0, math.copysign(width/2, self.unicycle_roll_angle)), transform(0, (math.copysign(
        #     width/2, self.unicycle_roll_angle) + math.sin(self.unicycle_roll_angle) * 1.5)), width=2)

        r_x = math.sin(self.unicycle_roll_angle) * (self.wheel_radius * 2) \
            + math.sin(self.unicycle_roll_angle + self.rider_lean_angle) * \
            (self.rider_height/2 + self.rider_offset_y)
        pygame.draw.line(surface, (255, 0, 0), transform(
            0, 0), transform(0, r_x), width=2)

    def act(self, action, delta_time):
        # Take in action (maps to keyboard presses in game),
        # update rider lean angle, wheel pitch momentum

        # type UnicycleInput = {
        #    leaning: -1 | 0 | 1,
        #    pedaling: -1 | 0 | 1 (Technically can use shift to inclue -2 and 2, but ignore for now)
        # }

        # action: [leaning, pedaling]

        # Assume that leaning/pedaling in action are already discretized
        leaning = action[0]
        pedaling = action[1]

        self.wheel_angular_momentum += pedaling * \
            self.pedaling_acceleration * delta_time

        dtheta = leaning * self.lean_speed * delta_time
        self.rider_lean_angle -= dtheta
        self.rider_lean_angle = np.clip(
            self.rider_lean_angle, -math.pi/2, math.pi/2)

    def update(self, delta_time):
        # Perform physics update step, according to delta time

        # Update roll from leaning
        r_x = math.sin(self.unicycle_roll_angle) * (self.wheel_radius * 2) \
            + math.sin(self.unicycle_roll_angle + self.rider_lean_angle) * \
            (self.rider_height/2 + self.rider_offset_y)
        torque = r_x * self.gravity * self.wheel_mass

        self.unicycle_roll_momentum += torque/self.wheel_inertia * delta_time

        self.unicycle_roll_angle += self.unicycle_roll_momentum * delta_time
        self.unicycle_roll_angle = np.clip(
            self.unicycle_roll_angle, -math.pi/2, math.pi/2)

        # Update wheel rotation
        dx = self.wheel_angular_momentum * self.wheel_radius * delta_time
        self.wheel_angular_momentum -= dx * self.friction

        # Update heading according to curvature
        denom = lerp(0, 1, invlerp(
            0, math.pi/2, abs(self.unicycle_roll_angle)))
        k = math.inf if abs(denom) < 1e-3 else 1/denom

        dtheta = self.wheel_angular_momentum/k * delta_time
        self.heading_angle -= math.copysign(dtheta, -self.unicycle_roll_angle)

        self.heading[0] = math.cos(self.heading_angle)
        self.heading[1] = math.sin(self.heading_angle)

        # Move according to heading
        self.position[0] += self.heading[0] * dx
        self.position[1] += self.heading[1] * dx

    def step(self, action, delta_time):
        self.act(action, delta_time)
        self.update(delta_time)


# Abstract all the drawing logic for
# visualizing episodes to this function,
# to then be used in following cells.
def draw(canvas: pygame.Surface, entities):
    canvas.fill((255, 255, 255, 255))

    for entity in entities:
        entity.draw(canvas)
