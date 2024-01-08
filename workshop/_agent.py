from vpython import *
import random
import numpy as np
import math

# Values taken from game

inner_radius = 5
outer_radius = 15

rider_height = 4
rider_offset_y = 0.6

wheel_radius = 1.1
wheel_mass = 5
wheel_inertia = 70
gravity = 9.81
friction = 0.9

class Loop:
    def __init__(self, inner_radius, outer_radius):
        self.track = extrusion(path=[vec(0,0,0), vec(0,-0.1, 0)],
                               color=color.gray(0.85),
                               shape=[ shapes.circle(radius=outer_radius),
                                       shapes.circle(radius=inner_radius) ])
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def generate_starting_pos(self):
        r = random.randrange(self.inner_radius, self.outer_radius)
        return vector(r, 0, 0)

def lerp(a, b, t):
    return t*b + (1-t)*a

def invlerp(a, b, c):
    return (c-a)/(b-a)


class Unicycle:
    def __init__(self, position):
        self.position = position
        self.length = wheel_radius * 2
        self.heading = vector(0, 0, -1)
        self.wheel = box(pos=self.position + vector(0, self.length/2, 0),
                         length=self.length,
                         height=self.length,
                         width=0.2,
                         color=color.green)

        self.rider_length = 0.5
        self.rider = box(pos=self.position + vector(0, self.length + rider_offset_y + rider_height/2, 0),
                         length=self.rider_length,
                         height=rider_height,
                         width=0.2,
                         color=color.red)

        self.heading_angle = radians(45)
        self.rider_lean_angle = 0
        self.unicycle_lean_angle = 0
        self.unicycle_lean_momentum = 0
        self.wheel_pitch_momentum = 0
        self.wheel_pitch = 0
        

        # self.wheel.axis = self.heading

    def update(self, tick_delta):
        # Update roll from leaning
        r_x = sin(self.unicycle_lean_angle) * (wheel_radius * 2) \
            + sin(self.rider_lean_angle) * (rider_height/2 + rider_offset_y)
        torque = r_x * gravity * wheel_mass

        self.unicycle_lean_momentum += torque/wheel_inertia * tick_delta

        self.unicycle_lean_angle += self.unicycle_lean_momentum * tick_delta

        # Update wheel rotation
        dx = self.wheel_pitch_momentum * wheel_radius * tick_delta
        self.wheel_pitch_momentum -= dx * friction

        self.wheel_pitch += self.wheel_pitch_momentum * tick_delta

        # Update heading according to curvature
        denom = lerp(0, 1, invlerp(0, pi/2, abs(self.unicycle_lean_angle)))
        k = inf if abs(denom) < 1e-6 else 1/denom

        self.heading_angle -= math.signum(self.unicycle_lean_angle) * self.wheel_omega/k * tick_delta

        # Update heading
        self.heading.x = 0
        self.heading.y = 0
        self.heading.z = -1
        self.heading.rotate_in_place(angle=self.heading_angle, axis=vector(0, 1, 0))
        self.wheel.axis = self.heading * self.length
        self.rider.axis = self.heading * self.rider_length

        # Move according to heading
        self.position += self.heading * self.wheel_pitch_momentum * tick_delta

scene.camera.pos = vector(0, 5, 10)
scene.camera.rotate(angle=radians(-30), axis=vector(1, 0, 0))

loop = Loop(inner_radius, outer_radius)
pos = loop.generate_starting_pos()
unicycle = Unicycle(pos)

while True:
    rate(100)
    unicycle.update(1/100)