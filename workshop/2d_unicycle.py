# Dumped from notebook
import numpy as np
import math
from time import sleep
from ipycanvas import Canvas, hold_canvas
import random
from ipywidgets import Output

screen_width = 400
screen_height = 300

world_width = 40
world_height = screen_height/screen_width * world_width

canvas = Canvas(width=screen_width, height=screen_height)
display(canvas)

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

pedaling_acceleration = 3 # radians
lean_speed = math.radians(200)



# for k in range(100):
#     agents = [(Agent(), Unicycle()), (Agent(), Unicycle()), (Agent(), Unicycle())] # Diff params
#     for agent, unicycle in agents:
#         state = unicycle.get_state()
#         action = agent.act(state)
#         unicycle.step(action, delta_time)

def lerp(a, b, t):
    return t*b + (1-t)*a

def invlerp(a, b, c):
    return (c-a)/(b-a)

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
        
    def draw_canvas(self, canvas):
        canvas.save()
        
        canvas.begin_path()
        canvas.fill_style = '#aaa'
        canvas.arc(0, 0, self.outer_radius, 0, 2 * math.pi, False)
        canvas.arc(0, 0, self.inner_radius, 0, 2 * math.pi, True)
        canvas.fill()
        
        canvas.begin_path()
        canvas.stroke_style = '#000'
        canvas.line_width = 0.1
        canvas.move_to(self.inner_radius, 0)
        canvas.line_to(self.outer_radius, 0)
        canvas.stroke()
        
        canvas.restore()

class Unicycle:
    def __init__(self, position, track):
        self.track = track
        self.position = position
        self.heading_angle = math.pi/2
        self.heading = [math.cos(self.heading_angle), math.sin(self.heading_angle)]
        
        # Unicycle state (radians)
        self.rider_lean_angle       = 0 # Lean of rider wrt front of unicycle
        self.unicycle_roll_angle    = 0 # Lean of unicycle wrt track
        self.unicycle_roll_momentum = 0 # Angular momentum of unicycle
        self.wheel_angular_momentum = 0
        
    def draw_canvas(self, canvas):
        canvas.save()
        
        length = 2
        width = 1
        

        canvas.translate(self.position[0], self.position[1])
        canvas.rotate(self.heading_angle)

        # Osculating circle
        if abs(self.unicycle_roll_angle) > 1e-3:
            canvas.begin_path()
            denom = lerp(0, 1, invlerp(0, math.pi/2, abs(self.unicycle_roll_angle)))
            k = math.inf if abs(denom) < 1e-6 else 1/denom
            canvas.begin_path()
            canvas.line_width = 0.3
            canvas.stroke_style = '#ff0'
            canvas.arc(0, -math.copysign(k, self.unicycle_roll_angle), k, 0, 2 * math.pi)
            canvas.stroke()
            
        on_track = self.track.on_track(self.position)
            
        # "Wheel"/body
        canvas.fill_style = '#0f0' if on_track else '#f00'
        canvas.fill_rect(-length/2, -width/2, length, width) # Align w/ heading, so on x-axis
        
        # Heading arrow
        canvas.begin_path()
        canvas.stroke_style = '#ff0'
        canvas.line_width = 0.3

        canvas.move_to(length/2, 0)
        canvas.line_to(length/2 + 2, 0)
        canvas.stroke()
        
        # Wheel roll helper arrow
        canvas.begin_path()
        canvas.stroke_style = '#00f' if self.unicycle_roll_angle > 0 else '#f00'
        canvas.move_to(0, -math.copysign(width/2, self.unicycle_roll_angle))
        canvas.line_to(0, -(math.copysign(width/2, self.unicycle_roll_angle) + math.sin(self.unicycle_roll_angle) * 1.5))
        canvas.stroke()
        
        canvas.restore()
        
    def act(self, action, delta_time):
        # Take in action (maps to keyboard presses in game),
        # update rider lean angle, wheel pitch momentum

        # type UnicycleInput = {
        #    leaning: -1 | 0 | 1,
        #    pedaling: -2 | -1 | 0 | 1 | 2,
        # }
        
        # action: [leaning, pedaling]
        
        # Assume that leaning/pedaling in action are already discretized
        leaning = action[0]
        pedaling = action[1]
        
        self.wheel_angular_momentum += pedaling * pedaling_acceleration * delta_time

        dtheta = leaning * lean_speed * delta_time
        self.rider_lean_angle += dtheta

    def update(self, delta_time):
        # Perform physics update step, according to delta time
        
        # Update roll from leaning
        r_x = math.sin(self.unicycle_roll_angle) * (wheel_radius * 2) \
            + math.sin(self.unicycle_roll_angle + self.rider_lean_angle) * (rider_height/2 + rider_offset_y)
        torque = r_x * gravity * wheel_mass

        self.unicycle_roll_momentum += torque/wheel_inertia * delta_time

        self.unicycle_roll_angle += self.unicycle_roll_momentum * delta_time
        self.unicycle_roll_angle = np.clip(self.unicycle_roll_angle, -math.pi/2, math.pi/2)

        # Update wheel rotation
        dx = self.wheel_angular_momentum * wheel_radius * delta_time
        self.wheel_angular_momentum -= dx * friction

        # Update heading according to curvature
        denom = lerp(0, 1, invlerp(0, math.pi/2, abs(self.unicycle_roll_angle)))
        k = math.inf if abs(denom) < 1e-3 else 1/denom

        dtheta = self.wheel_angular_momentum/k * delta_time
        self.heading_angle -= math.copysign(dtheta, self.wheel_angular_momentum)

        self.heading[0] = math.cos(self.heading_angle)
        self.heading[1] = math.sin(self.heading_angle)
        
        # Move according to heading
        self.position[0] += self.heading[0] * dx
        self.position[1] += self.heading[1] * dx
        
    def step(self, action, delta_time):
        self.act(action, delta_time)
        self.update(delta_time)
        
track = Loop(inner_radius, outer_radius) 
position = track.generate_starting_position()
unicycle = Unicycle(position, track)

while True:
    delta_time = 1/60
    
    with hold_canvas():
        canvas.on_key_down(on_keyboard_event)

        canvas.clear()

        canvas.save()
        
        canvas.scale(screen_width/world_width, -screen_height/world_height)
        canvas.translate(world_width/2, -world_height/2)
        
        track.draw_canvas(canvas)
        unicycle.draw_canvas(canvas)
        
        action = [0, 1]
        unicycle.step(action, delta_time)
        
        canvas.restore()
        
    sleep(delta_time)