# file runs policy evaluation for infinite horizon nonlinear control with discounting

# specific example is ball suspension, Example 4-9-3 from automatic control systems
# the control is the voltage maintained

import numpy as np
import pandas as pd
import numpy.linalg as npla
import scipy.linalg as scila
import random
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from lstd_boost import *
from fitted_value import *

##########################################################

class ball_suspension():
    # object that represents the magnetic ball suspension, MDP system
    # state space is given by (height, velocity, current), all floats
    # height is how far the mass is below the electromagnet
    # velocity is defined w.r.t to the mass's position
    # current is floating through the circuit


    def __init__(self, R, L, M,  g = 9.8, time_discrete = 0.1, gamma = 0.99):
        # initializes the ball suspension system
        # R: float, representing resistance of resistor (Ohms)
        # L: float, inductance of inductor (H)
        # M: float, mass of ball (kg)
        # gamma: float, discount parameter 
        # g: float, gravitational acceleration (m/s^2)
        # time_discrete: float, discretization level of continuous time system

        # sets parameters
        self.resistance = R
        self.inductance = L
        self.mass = M
        self.gravity = g
        self.gamma = gamma
        self.time_discrete = time_discrete

        self.state_dim = 3
        self.action_dim = 1
        self.tau = 2 * np.pi


    def reset(self, height, velocity, current):
        # resets the system to the state given by the arguments
        self.height = height
        self.velocity = velocity
        self.current = current
        self.init_height = height


    def step(self, voltage):
        # takes a step on the Markov Decision Process, where the action
        # is the voltage inputted
        # returns
        # tuple of 3 floats representing the next state
        # cost: cost of current timestep

        # computes the cost
        cost = np.power(self.velocity, 2) + np.power(self.current, 2)

        # takes a step
        curr_height = self.height
        curr_vel = self.velocity
        curr_current = self.current

        self.height = curr_height + self.time_discrete * curr_vel
        self.velocity = curr_vel + self.time_discrete \
                                    * (self.gravity - np.power(curr_current, 2) / (self.mass * curr_height))
        self.current = curr_current + self.time_discrete \
                                    * (-self.resistance / self.inductance * curr_current \
                                        + voltage / self.inductance)

        return (self.height, self.velocity, self.current), cost


    def state(self):
        return (self.height, self.velocity, self.current)



##########################################################

class ball_suspension_controller():
    # object that represents the ball_suspension system with controller
    # same as ball_suspension


    def __init__(self, R, L, M, g = 9.8, gamma = 0.99):
        # initializes the ball suspension system
        # R: float, representing resistance of resistor (Ohms)
        # L: float, inductance of inductor (H)
        # M: float, mass of ball (kg)
        # gamma: float, discount parameter 
        # g: float, gravitational acceleration (m/s^2)
        # time_discrete: float, discretization level of continuous time system

        # sets parameters
        self.resistance = R
        self.inductance = L
        self.mass = M
        self.gravity = g
        self.gamma = gamma

        self.tau = 2 * np.pi


    def policy(self, height, velocity, current):
        # returns the voltage that should be outputted given the current state
        # height: float, denotes current height of ball
        # velocity: float, denotes current velocity of ball
        # current: float, denotes current flowing through wire
        #
        # returns
        # voltage: float, denoting control voltage to be applied

        pass




class oscillation(ball_suspension_controller):

    def __init__(self, R, L, M, init_height, g = 9.8, gamma = 0.99):
        self.init_height = init_height
        super(oscillation, self).__init__(R, L, M, g, gamma)


    def policy(self, height, velocity, current):
        # returns the voltage that should be outputted given the current state
        # height: float, denotes current height of ball
        # velocity: float, denotes current velocity of ball
        # current: float, denotes current flowing through wire
        #
        # returns
        # voltage: float, denoting control voltage to be applied

        term_one = self.resistance * current
        term_two = self.mass * velocity / (2 * current)
        term_three = self.gravity + np.power(self.tau, 2) * (2 * height - self.init_height)

        voltage = term_one + term_two * term_three

        return voltage




##########################################################

def test_stable():
    # checks if the closed form solution analytically is stable
    # basic bug check

    mass = 1.
    gravity = 9.8
    inductance = 1.
    resistance = 10.

    height = 1.
    velocity = 0.
    current = np.sqrt(gravity * mass * height)

    voltage = resistance * current

    system = ball_suspension(resistance, inductance, mass, gravity)

    system.reset(height, velocity, current)

    for i in range(10):
        system.step(voltage)
        print(system.state()) # should be all the same to represent steady state



def test_other():
    # control law I derived for oscillation

    mass = 20.
    gravity = 9.8
    inductance = 1.
    resistance = 5.

    tau = 2. * np.pi
    C = np.power(1 / tau, 2) 
    init_height = 30.
    init_velocity = init_height * tau * C
    init_current = np.sqrt(gravity * mass * init_height)

    system = ball_suspension(resistance, inductance, mass, gravity, time_discrete = 0.0001)
    controller = oscillation(resistance, inductance, mass, gravity, init_height)
    system.reset(init_height, init_velocity, init_current)

    for i in range(10000000):
        voltage = controller.policy(*system.state())
        system.step(voltage)
        if i % 100000 == 0:
            print(voltage)
            print(system.state())

    



test_other()





                                    

