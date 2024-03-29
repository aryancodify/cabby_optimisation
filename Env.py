# Import routines

import numpy as np
import math
import random
from itertools import permutations

# Defining hyperparameters
m = 5  # number of cities, ranges from 0 ..... m-1
t = 24  # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5  # Per hour fuel and other costs
R = 9  # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(0, 0)] + \
            list(permutations([i for i in range(m)], 2))
        self.state_space = [[x, y, z]
                            for x in range(m) for y in range(t) for z in range(d)]
        self.state_init = random.choice(self.state_space)
        #self.state_init = [0,0,0]
        # Start the first round
        self.reset()

    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. 
        This method converts a given state into a vector format. 
        Hint: The vector is of size m + t + d."""

        state_encod = [0 for _ in range(m+t+d)]
        state_encod[self.state_get_loc(state)] = 1
        state_encod[m+self.state_get_time(state)] = 1
        state_encod[m+t+self.state_get_day(state)] = 1

        return state_encod

    # Use this function if you are using architecture-2

    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. 
        This method converts a given state-action pair into a vector format. 
        Hint: The vector is of size m + t + d + m + m."""
        state_encod = [0 for _ in range(m+t+d+m+m)]
        state_encod[self.state_get_loc(state)] = 1
        state_encod[m+self.state_get_time(state)] = 1
        state_encod[m+t+self.state_get_day(state)] = 1
        if (action[0] != 0):
            state_encod[m+t+d+self.action_get_pickup(action)] = 1
        if (action[1] != 0):
            state_encod[m+t+d+m+self.action_get_drop(action)] = 1

        return state_encod

    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)

        if requests > 15:
            requests = 15
        # (0,0) is not considered as customer request, however the driver is free to refuse all
        # customer requests. Hence, add the index of action (0,0).
        possible_actions_index = random.sample(range(1, (m-1)*m + 1), requests) + [0]
        actions = [self.action_space[i] for i in possible_actions_index]

        return possible_actions_index, actions

    def update_time_day(self, state, time):
        """
        Takes in the current state and time taken for driver's journey to return
        the state post that journey.
        """
        time = int(time)
        state_time = self.state_get_time(state)
        state_day = self.state_get_day(state)
        if (state_time + time) < 24:
            state_time = state_time + time
            return state_time, state_day
        else:
            # if time is > 24 we increment the day by 1.
            state_time = state_time + time - 24
            if state_day == 6:
                state_day = 0
            else:
                state_day = state_day + 1
            return state_time, state_day

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        next_state = []
        total_time = 0
        if ((self.action_get_pickup(action) == 0) and (self.action_get_drop(action) == 0)):
            #print("REFUSE")
            #print(state)
            state_time, state_day = self.update_time_day(state, 1)
            #print(state_time, state_day)
            total_time += 1
            next_state = [state[0], state_time, state_day]
            return next_state, total_time
        # means driver is already at pickup point
        if self.state_get_loc(state) == self.action_get_pickup(action):
            time = Time_matrix[self.state_get_loc(state)][self.action_get_drop(
                action)][self.state_get_time(state)][self.state_get_day(state)]
            next_state = [state[0], state[1], state[2]]
            total_time += time
        else:
            # Driver is not at the pickup point, he needs to travel to pickup point first and then drop the passenger.
            #time take to reach pickup point
            time = Time_matrix[self.state_get_loc(state)][self.action_get_pickup(
                action)][self.state_get_time(state)][self.state_get_day(state)]
            total_time += time
            state_time, state_day = self.update_time_day(state, time)
            # the driver is now at the pickup point
            next_state = [action[0], state_time, state_day]
            #Time taken to drop the passenger
            time = Time_matrix[self.action_get_pickup(
                action)][self.action_get_drop(action)][state_time][state_day]
            total_time += time
        state_time, state_day = self.update_time_day(next_state, time)
        next_state = [self.action_get_drop(action), state_time, state_day]
        return next_state, total_time

    def reset(self):
        """Return the current state and action space"""
        return self.action_space, self.state_space, self.state_init

    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        # when driver chooses to stay idle
        if ((self.action_get_pickup(action) == 0) and (self.action_get_drop(action) == 0)):
            reward = - C

        elif (self.state_get_loc(state) == self.action_get_pickup(action)):
            #passenger_time is when passenger is in the cab and this results in both revenue and battery cost
            #idle_time is the time taken for the driver to reach the passenger pickup point, this results only in battery cost.
            passenger_time = Time_matrix[self.state_get_loc(state)][self.action_get_drop(
                action)][self.state_get_time(state)][self.state_get_day(state)]
            idle_time = 0
            reward = (R * passenger_time) - (C * (passenger_time + idle_time))

        else:
             #passenger_time is when passenger is in the cab and this results in both revenue and battery cost
            #idle_time is the time taken for the driver to reach the passenger pickup point, this results only in battery cost.
            idle_time = Time_matrix[self.state_get_loc(state)][self.action_get_pickup(
                action)][self.state_get_time(state)][self.state_get_day(state)]
            state_time, state_day = self.update_time_day(state, idle_time)
            passenger_time = Time_matrix[self.action_get_pickup(
                action)][self.action_get_drop(action)][state_time][state_day]
            reward = (R * passenger_time) - (C * (passenger_time + idle_time))

        return reward

    def step(self, state, action, Time_matrix):
        """
        Take a trip as cabby to get rewards next step and total time spent
        """
        next_state, total_time = self.next_state_func(
            state, action, Time_matrix)
        return self.reward_func(state, action, Time_matrix), next_state, total_time

    def state_get_loc(self, state):
        return state[0]

    def state_get_time(self, state):
        return state[1]

    def state_get_day(self, state):
        return state[2]

    def action_get_pickup(self, action):
        return action[0]

    def action_get_drop(self, action):
        return action[1]

    def state_set_loc(self, state, loc):
        state[0] = loc

    def state_set_time(self, state, time):
        state[1] = time

    def state_set_day(self, state, day):
        state[2] = day

    def action_set_pickup(self, action, pickup):
        action[0] = pickup

    def action_set_drop(self, action, drop):
        action[1] = drop
