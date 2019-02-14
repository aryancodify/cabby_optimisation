# Import routines

import numpy as np
import math
import random
from itertools import permutations 

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space =  [(0,0)] + list(permutations([i for i in range(1,m+1)], 2))
        self.state_space = [[x,y,z] for x in range(1,m+1) for y in range(t) for z in range(d)]
        self.state_init = [1]

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = [0 for _ in range(m+t+d)]
        state_encod[state[0]-1] = 1
        state_encod[m+state[1]] = 1
        state_encod[m+t+state[2]] = 1
            
        return state_encod


    # Use this function if you are using architecture-2 
    def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        state_encod = [0 for _ in range(m+t+d+m+m)]
        
        state_encod[state[0]-1] = 1
        state_encod[m+state[1]] = 1
        state_encod[m+t+state[2]] = 1
        if (action[0] != 0 ) : 
            state_encod[m+t+d+action[0]-1] = 1
        if (action[1] != 0 ) : 
            state_encod[m+t+d+m+action[1]-1] = 1
        
        return state_encod

    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 1:
            requests = np.random.poisson(2)
        if location == 2:
            requests = np.random.poisson(12)
        if location == 3:
            requests = np.random.poisson(4)
        if location == 4:
            requests = np.random.poisson(7)
        if location == 5:
            requests = np.random.poisson(8)
            
        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        actions.append([0,0])

        return possible_actions_index,actions  
    
    def update_time_day(self, state, time):
        if (state[1] + time) < 24: 
            state[1] = state[1] + time
            return state
        else: #if time is > 24 we increment the day by 1.
            state[1] = state[1] + time - 24
            if state[2] == 6:
                state[2] = 0
            else : 
                state[2] = state[2] + 1
            return state

    def next_state_func(self, state, action, Time_matrix):
        
        """Takes state and action as input and returns next state"""
        if state[0] == action[0] : #means driver is already at pickup point
            time = Time_matrix[state[0]][action[1]][state[1]][state[2]]
       
        else: #Driver is not at the pickup point, he needs to travel to pickup point first and then drop the passenger.
            time = Time_matrix[state[0]][action[0]][state[1]][state[2]] #time take to reach pickup point
            state = self.update_time_day(state, time)
            
            state[0] = action[0] #the driver is not at the pickup point
            time = Time_matrix[state[0]][action[1]][int(state[1])][state[2]] #Time taken to drop the passenger          
        next_state = [action[1], int(state[1]+time), state[2]]
        return next_state

    def reset(self):
        return self.action_space, self.state_space, self.state_init



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        if ((action[0] == 0) and (action[1] == 0)): #when driver chooses to stay idle
            reward = - C    
            
        elif (state[0] == action[0]):
            #passenger_time is when passenger is in the cab and this results in both revenue and battery cost
            #idle_time is the time taken for the driver to reach the passenger pickup point, this results only in battery cost.
            passenger_time = Time_matrix[state[0]][action[1]][state[1]][state[2]]
            idle_time = 0
            reward = (R * passenger_time) - (C *(passenger_time + idle_time))
            
        else:
             #passenger_time is when passenger is in the cab and this results in both revenue and battery cost
            #idle_time is the time taken for the driver to reach the passenger pickup point, this results only in battery cost.
            idle_time = Time_matrix[state[0]][action[0]][state[1]][state[2]]
            state = self.update_time_day(state, idle_time)
            state[0] = action[0]
            passenger_time = Time_matrix[state[0]][action[1]][int(state[1])][state[2]]          
            reward = (R * passenger_time) - (C *(passenger_time + idle_time))
        
        return reward




    




