"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the FaultyBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon, fault): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)
"""

import numpy as np
import math

# START EDITING HERE
# You can use this space to define any helper functions that you need

def KL(p, q):
    if p==q:
        return 0
    if q==0 or q==1:
        return float("inf")
    if p==1:
        return -math.log(q)
    if p==0:
        return -math.log(1-q)
    return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))

def ucb_solver(time, action, count, c=2):
    if count==0:
        return 1
    return min(action + math.sqrt(c*math.log(time)/count), 1)

def klucb_rhs(time, count, c=0):
    return (math.log(time)+c*math.log(math.log(time)))/count


def klucb_solver(time, action, count, c=0):
    if count==0:
        return 1
    
    bound = klucb_rhs(time, count, c)
    low, high = action, 1

    while high-low>=1e-4:
        mid = (low + high) / 2
        div = KL(action, mid)
        if div<=bound and bound-div<=1e-4:
            return mid
        elif div > bound:
            high = mid
        else:
            low = mid
    return low


# END EDITING HERE

class FaultyBanditsAlgo:
    def __init__(self, num_arms, horizon, fault):
        # You can add any other variables you need here
        self.num_arms = num_arms
        self.horizon = horizon
        self.fault = fault # probability that the bandit returns a faulty pull
        # START EDITING HERE
        self.eps = 0.1
        self.time = 1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.ucb = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # self.time += 1
        # if np.random.random() < self.eps:
        #     return np.random.randint(self.num_arms)
        # else:
        #     return np.argmax(self.values)
        if 0 in self.counts:
                for arm, count in enumerate(self.counts):
                    if count==0:
                        return arm
        return np.argmax(self.ucb)

        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.time += 1
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        corrected_reward = max((reward - self.fault/2)/(1 - self.fault), 0)
        self.values[arm_index] = ((n - 1) * value + corrected_reward)/n
        self.ucb = [klucb_solver(self.time, self.values[index], self.counts[index])
                    for index in range(self.num_arms)]
        #END EDITING HERE

