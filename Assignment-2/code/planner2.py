# import argparse
# import sys
# import os
# import numpy as np
# from pulp import *


# class MDPPlanning():
#     def __init__(self, path):
#         mdp_data = open(path)

#         # Populating the MDP data

#         lines = mdp_data.readlines()
#         self.nums_states = int(lines[0].split()[-1])
#         self.nums_actions = int(lines[1].split()[-1])
#         self.mdp_rewards = np.zeros(
#             shape=(self.nums_states, self.nums_actions, self.nums_states))
#         self.mdp_transitions = np.zeros(
#             shape=(self.nums_states, self.nums_actions, self.nums_states))
#         self.discount_factor = float(lines[-1].split()[-1])
#         self.type = lines[-2].split()[-1]
#         self.end_states = np.array(list(map(int, lines[2].split()[1:])))
#         self.value_function = np.zeros(shape=(self.nums_states))
#         self.policy = np.zeros(shape=(self.nums_states))
#         # Populating the rewards and transitions matrix

#         for lines in lines[3:-2]:
#             lines = lines.split()
#             self.mdp_rewards[int(lines[1])][int(lines[2])][int(lines[3])] = float(
#                 lines[4])
#             self.mdp_transitions[int(lines[1])][int(lines[2])][int(lines[3])] = float(
#                 lines[5])

#         mdp_data.close()

#         # End of constructor

#     def value_iteration(self, tolerance=1e-12):
#         """ Vectorized value iteration algorithm """
#         t=0
#         while True:
#             t+=1
#             error = 0
#             action_matrix = np.max(np.sum(self.mdp_transitions * (
#                 self.mdp_rewards + self.discount_factor * self.value_function), axis=2), axis=1)
#             error = np.max(np.abs(action_matrix - self.value_function))
#             self.value_function = action_matrix
            
#             if (error < tolerance):
#                 break
#         print(t)

#     def howards_policy_iteration(self):
#         """ Howard's policy iteration algorithm """

#         # Starting with a random policy
#         self.policy = np.random.randint(
#             low=0, high=self.nums_actions, size=self.nums_states)

#         # Evalutates the random policy
#         self.policy_evaluation()

#         # Evaluates the best action for each state
#         best_action_matrix = np.argmax(np.sum(self.mdp_transitions * (
#             self.mdp_rewards + self.discount_factor * self.value_function), axis=2), axis=1)

#         # Iterates until the policy converges
#         while not np.array_equal(self.policy, best_action_matrix):
#             self.policy = best_action_matrix
#             self.policy_evaluation()
#             best_action_matrix = np.argmax(np.sum(self.mdp_transitions * (
#                 self.mdp_rewards + self.discount_factor * self.value_function), axis=2), axis=1)
#         pass

#     def linear_programming(self):
#         """ Linear programming algorithm """

#         # Defining the problem
#         mdp = LpProblem("MDP", LpMinimize)

#         # Defining the variables
#         value_function = LpVariable.dicts(
#             "state", range(self.nums_states), lowBound=0)

#         # Defining the objective function
#         mdp += (lpSum([value_function[state]
#                        for state in range(self.nums_states)]))

#         # Defining the constraints
#         for state in range(self.nums_states):
#             for action in range(self.nums_actions):
#                 mdp += lpSum([self.mdp_transitions[state][action][next_state] * (self.mdp_rewards[state][action][next_state] +
#                                                                                  self.discount_factor * value_function[next_state]) for next_state in range(self.nums_states)]) <= value_function[state]

#                 # Solving the linear programming problem
#         mdp.solve(PULP_CBC_CMD(msg=0))

#         for state in range(self.nums_states):
#             self.value_function[state] = value_function[state].varValue

#             # Finding the best policy
#         self.policy = np.argmax(np.sum(self.mdp_transitions * (
#             self.mdp_rewards + self.discount_factor * self.value_function), axis=2), axis=1)

#     def policy_evaluation(self, policy_path=None, tolerance=1e-12):
#         """ Policy evaluation algorithm """

#         if policy_path != None:
#             with open(policy_path) as file:
#                 for i, line in enumerate(file):
#                     self.policy[i] = int(line)

#         self.value_function = np.zeros(self.nums_states)
#         while True:
#             error = 0
#             value_matrix = np.zeros(self.nums_states)

#             for state in range(self.nums_states):
#                 optimal_action = self.policy[state]
#                 value_matrix[state] = np.sum(self.mdp_transitions[state, int(optimal_action), :] * (
#                     self.mdp_rewards[state, int(optimal_action), :] + self.discount_factor * self.value_function))

#             error = np.max(np.abs(self.value_function - value_matrix))
#             self.value_function = value_matrix
#             if (error < tolerance):
#                 break

#     def print_output(self):
#         for i in range(self.nums_states):
#             print(str(self.value_function[i]) +
#                   " " + str(self.policy[i]))


# def mdp_path(string):
#     if os.path.isfile(string):
#         # print("MDP Path used %s" % string)
#         return string
#     else:
#         raise NotADirectoryError(string)


# def policy_path(string):
#     if os.path.isfile(string):
#         # print("Policy being evaluated %s" % string)
#         return string
#     else:
#         raise NotADirectoryError(string)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mdp", type=mdp_path)
#     parser.add_argument("--algorithm", type=str, default="vi")
#     parser.add_argument("--policy", type=policy_path)

#     args = parser.parse_args()

#     mdp = MDPPlanning(args.mdp)

#     # print(mdp.nums_actions, mdp.nums_states,
#     #       mdp.mdp_rewards[299][5][301], mdp.mdp_transitions[0][0][300], mdp.end_states)

#     if (args.policy == None):
#         # Computing optimal policy
#         if not (args.algorithm == "hpi" or args.algorithm == "vi" or args.algorithm == "lp"):
#             print("Algorithm should be hpi, vi or lp")
#             sys.exit(0)
#         elif args.algorithm == "vi":
#             mdp.value_iteration()
#             mdp.print_output()

#         elif args.algorithm == "hpi":
#             mdp.howards_policy_iteration()
#             mdp.print_output()

#         else:
#             mdp.linear_programming()
#             mdp.print_output()
#             # print("%s Algorithm used" % args.algorithm)

#     else:
#         pass
#         # Evaluating the given policy
#         mdp.policy_evaluation(args.policy)
#         mdp.print_output()



"""

This is some code 
"""
#!/usr/bin/python

from argparse import ArgumentParser as ap
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, value, LpStatusOptimal
from sys import maxsize

class State:
    
    def __init__(self, i):
        self.i = i # state number
        self.A = set() # actions that can be taken

class myMDP:
    
    def __init__(self, S, A, e):
        self.ns = S # number of states
        self.na = A # number of actions
        self.S = {i:State(i) for i in range(S)} # all states
        self.A = [i for i in range(A)] # all actions
        self.e = e # end states
        self.T = [[[]for _ in range(A)] for _ in range(S)] # transitions
        self.thres = 1e-10 # threshold to stop
    
    def setParams(self, g, t):
        self.g = g # gamma
        self.t = t # type
    
    def policyEvaluation(self, policy):
        modulo = self.ns
        if len(policy) != self.ns:
            modulo = len(policy)
        V = [0 for _ in range(self.ns)] # value function
        curr = None # copy of value function
        while True:
            curr = V.copy()
            for s1 in self.S: # iterate over states and find V
                if s1 in self.e: # end state
                    V[s1] = 0
                    continue
                Vs1 = 0
                trans = self.T[s1][policy[s1 % modulo]] # get transition data
                for (s2,r,p) in trans: # compute value
                    if s2 not in self.e:
                        Vs1 += p * (r + self.g * curr[s2])
                    else:
                        Vs1 += p * r
                V[s1] = Vs1
            if np.linalg.norm(np.array(curr) - np.array(V)) <= self.thres: # check threshold
                return V, policy
    
    def valueIteration(self):
        V = [0 for _ in range(self.ns)] # value function
        curr = None # copy of value function
        policy = [0 for _ in range(self.ns)] # policy
        while True:
            curr = V.copy()
            for s1 in self.S: # iterate over states and find V
                if s1 in self.e: # end state
                    V[s1] = 0
                    continue
                optimal = 0
                maxV = -maxsize
                for a in sorted(self.S[s1].A): # iterate over actions
                    trans = self.T[s1][a] # get transition data
                    Vs1 = 0
                    for (s2,r,p) in trans: # compute value
                        if s2 not in self.e:
                            Vs1 += p * (r + self.g * curr[s2])
                        else:
                            Vs1 += p * r
                    if Vs1 > maxV:
                        maxV = Vs1
                        optimal = a
                if len(self.S[s1].A) == 0:
                    maxV = 0
                V[s1] = maxV # set value
                policy[s1] = optimal # set policy
            if np.linalg.norm(np.array(curr) - np.array(V)) <= self.thres: # check threshold
                for s in self.S: # iterate over states and check/assign policy
                    if len(self.S[s].A) == 0 or (policy[s] in self.S[s].A):
                        continue
                    for a in self.S[s].A:
                        policy[s] = a
                        break
                return V, policy
    
    def howardsPolicyIteration(self):
        policy = [] # policy
        for s in self.S: # iterate over states and get initial policy
            if len(self.S[s].A) == 0:
                policy.append(0)
            else:
                for a in sorted(self.S[s].A):
                    policy.append(a)
                    break
        i = 0
        while i < 10: # 10 improvement iterations
            i += 1
            V, policy = self.policyEvaluation(policy)
            policy1 = policy.copy() # check policy
            for s1 in self.S: # iterate over states and find Q
                if s1 in self.e: # end state
                    continue
                currQ = V[s1]
                maxQ = currQ
                optimal = policy[s1]
                for a in sorted(self.S[s1].A): # iterate over actions
                    Qs1a = 0
                    trans = self.T[s1][a] # get transition data
                    for (s2,r,p) in trans: # compute value
                        if s2 not in self.e:
                            Qs1a += p * (r + self.g * V[s2])
                        else:
                            Qs1a += p * r
                    if abs(Qs1a - currQ) >= self.thres and Qs1a > maxQ:
                        optimal = a
                        maxQ = Qs1a
                policy1[s1] = optimal # set optimal
            if np.array_equal(policy,policy1): # reached optimal
                for s in self.S: # iterate over states and check/assign policy
                    if len(self.S[s].A) == 0 or (policy[s] in self.S[s].A):
                        continue
                    for a in self.S[s].A:
                        policy[s] = a
                        break
                return V, policy
            policy = policy1 # use improved policy
        return V, policy
    
    def linearProgramming(self):
        prob = LpProblem('V',LpMinimize) # problem
        V = [LpVariable('V' + str(i)) for i in range(self.ns)] # value function
        prob += lpSum(V) # value function constraint
        for s1 in self.S: # iterate over states and add constraints
            for a in self.S[s1].A: # iterate over actions
                trans = self.T[s1][a] # get transition data
                Qsa = []
                for (s2,r,p) in trans: # compute value
                    if s2 not in self.e:
                        Qsa.append(p * (r + self.g * V[s2]))
                    else:
                        Qsa.append(p * r)
                if len(trans) != 0:
                    prob += V[s1] >= lpSum(Qsa) # constraint for optimal
            if len(self.S[s1].A) == 0:
                prob += V[s1] >= 0 # non-negative constraint
        res = prob.solve(PULP_CBC_CMD(msg = 0)) # solve LP
        # assert res == LpStatusOptimal # assert optimality
        Vf = [value(x) for x in V]
        policy = [0 for _ in range(self.ns)]
        for s1 in self.S: # iterate over states to assign policy
            Qs = [-maxsize for _ in range(self.na)]
            for a in self.S[s1].A: # iterate over actions
                trans = self.T[s1][a] # get transition data
                Qs[a] = 0
                for (s2,r,p) in trans: # compute value
                    if s2 not in self.e:
                        Qs[a] += p * (r + self.g * Vf[s2])
                    else:
                        Qs[a] += p * r
            policy[s1] = np.argmax(Qs) # set policy
        for s in self.S: # iterate over states and check/assign policy
            if len(self.S[s].A) == 0 or (policy[s] in self.S[s].A):
                continue
            for a in self.S[s].A:
                policy[s] = a
                break
        return Vf, policy
    
    def solver(self, policy, algo):
        if len(policy) != 0: # policy file given
            return self.policyEvaluation(policy)
        else:
            if algo == 'vi': # value iteration
                return self.valueIteration()
            elif algo == 'hpi': # Howard's policy iteration
                return self.howardsPolicyIteration()
            elif algo == 'lp': # linear programming
                return self.linearProgramming()

parser = ap()
parser.add_argument('--mdp', required = True, type = str, help = 'Path to MDP file')
parser.add_argument('--algorithm', required = False, type = str, help = 'Algorithm to use', default = 'lp')
parser.add_argument('--policy', required = False, type = str, help = 'Policy file', default = '')

args = parser.parse_args()
mdpfile = args.mdp
algo = args.algorithm
policyfile = args.policy

# create MDP and states
with open(mdpfile,'r') as f:
    ns = int(f.readline().split()[1])
    na = int(f.readline().split()[1])
    end = list(map(int,f.readline().split()[1:]))
    M = myMDP(ns,na,end)
    line = ''
    while True:
        line = f.readline().split()
        if line[0] == 'transition':
            s1, a, s2, r, p = list(map(int,line[1:4])) + list(map(float,line[4:]))
            M.S[s1].A.add(a)
            M.T[s1][a].append((s2,r,p))
        else:
            break
    t = line[1]
    disc = float(f.readline().split()[1])
    M.setParams(disc,t)

# create policy
policy = {}
cricketMap = {0:0,1:1,2:2,4:3,6:4}
if policyfile != '':
    with open(policyfile,'r') as f:
        lines = f.readlines()
        for i, a in enumerate(lines):
            if len(a.strip().split()) == 2:
                policy[i] = cricketMap[int(a.split()[1])]
            else:
                policy[i] = int(a)

# invoke solver
V, policy = M.solver(policy,algo)
for i in range(len(policy)):
    print(f'{V[i]:.6f} {policy[i]}')