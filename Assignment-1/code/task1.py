"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.

    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).

    - get_reward(self, arm_index, reward): This method is called just after the
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon

    def give_pull(self):
        raise NotImplementedError

    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm

class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)

    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)

    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

"""
Kullback–Leibler divergence
Reference: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

"""
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

"""
Belief distribution for Thompson Sampling

"""
def belief(success, failure):
    return np.random.beta(success+1, failure+1)


class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.time = 1                
        self.counts = np.zeros(num_arms)     
        self.values = np.ones(num_arms)
        self.ucb = np.ones(num_arms)

    def give_pull(self):
        # Initially pulling in Round Robin Fashion till each arm is pulled atleast once
        if 0 in self.counts:
            for arm, count in enumerate(self.counts):
                if count==0:
                    return arm
                
        return np.argmax(self.ucb)

    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        self.time += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        self.values[arm_index] = ((n - 1) * value + reward)/n
        self.ucb = [ucb_solver(self.time, self.values[i], self.counts[i]) 
                    for i in range(self.num_arms)]


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.time = 1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.ucb = np.zeros(num_arms)

    def give_pull(self):
        if 0 in self.counts:
                for arm, count in enumerate(self.counts):
                    if count==0:
                        return arm
        return np.argmax(self.ucb)


    def get_reward(self, arm_index, reward):
        self.time += 1
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        self.values[arm_index] = ((n - 1) * value + reward)/n
        self.ucb = [klucb_solver(self.time, self.values[index], self.counts[index])
                    for index in range(self.num_arms)]


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.success = np.zeros(num_arms)
        self.failure = np.zeros(num_arms)

    def give_pull(self):
        return np.argmax([belief(self.success[i], self.failure[i]) 
                          for i in range(self.num_arms)])

    def get_reward(self, arm_index, reward):
        if (reward == 1):
            self.success[arm_index] += 1
        else:
            self.failure[arm_index] += 1
