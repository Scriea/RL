import argparse
import sys
import os
import numpy as np
from pulp import *


class MDPPlanning():
    def __init__(self, path):
        mdp_data = open(path)

        # Populating the MDP data

        lines = mdp_data.readlines()
        self.nums_states = int(lines[0].split()[-1])
        self.nums_actions = int(lines[1].split()[-1])
        self.mdp_rewards = np.zeros(
            shape=(self.nums_states, self.nums_actions, self.nums_states))
        self.mdp_transitions = np.zeros(
            shape=(self.nums_states, self.nums_actions, self.nums_states))
        self.discount_factor = float(lines[-1].split()[-1])
        self.type = lines[-2].split()[-1]
        self.end_states = np.array(list(map(int, lines[2].split()[1:])))
        self.value_function = np.zeros(shape=(self.nums_states))
        self.policy = np.zeros(shape=(self.nums_states))
        # Populating the rewards and transitions matrix

        for lines in lines[3:-2]:
            lines = lines.split()
            self.mdp_rewards[int(lines[1])][int(lines[2])][int(lines[3])] = float(
                lines[4])
            self.mdp_transitions[int(lines[1])][int(lines[2])][int(lines[3])] = float(
                lines[5])

        mdp_data.close()

        # End of constructor

    def value_iteration(self, tolerance=1e-12):
        """ Vectorized value iteration algorithm """
        t=0
        while True:
            t+=1
            error = 0
            action_matrix = np.max(np.sum(self.mdp_transitions * (
                self.mdp_rewards + self.discount_factor * self.value_function), axis=2), axis=1)
            error = np.max(np.abs(action_matrix - self.value_function))
            self.value_function = action_matrix
            
            if (error < tolerance):
                break
        print(t)

    def howards_policy_iteration(self):
        """ Howard's policy iteration algorithm """

        # Starting with a random policy
        self.policy = np.random.randint(
            low=0, high=self.nums_actions, size=self.nums_states)

        # Evalutates the random policy
        self.policy_evaluation()

        # Evaluates the best action for each state
        best_action_matrix = np.argmax(np.sum(self.mdp_transitions * (
            self.mdp_rewards + self.discount_factor * self.value_function), axis=2), axis=1)

        # Iterates until the policy converges
        while not np.array_equal(self.policy, best_action_matrix):
            self.policy = best_action_matrix
            self.policy_evaluation()
            best_action_matrix = np.argmax(np.sum(self.mdp_transitions * (
                self.mdp_rewards + self.discount_factor * self.value_function), axis=2), axis=1)
        pass

    def linear_programming(self):
        """ Linear programming algorithm """

        # Defining the problem
        mdp = LpProblem("MDP", LpMinimize)

        # Defining the variables
        value_function = LpVariable.dicts(
            "state", range(self.nums_states), lowBound=0)

        # Defining the objective function
        mdp += (lpSum([value_function[state]
                       for state in range(self.nums_states)]))

        # Defining the constraints
        for state in range(self.nums_states):
            for action in range(self.nums_actions):
                mdp += lpSum([self.mdp_transitions[state][action][next_state] * (self.mdp_rewards[state][action][next_state] +
                                                                                 self.discount_factor * value_function[next_state]) for next_state in range(self.nums_states)]) <= value_function[state]

                # Solving the linear programming problem
        mdp.solve(PULP_CBC_CMD(msg=0))

        for state in range(self.nums_states):
            self.value_function[state] = value_function[state].varValue

            # Finding the best policy
        self.policy = np.argmax(np.sum(self.mdp_transitions * (
            self.mdp_rewards + self.discount_factor * self.value_function), axis=2), axis=1)

    def policy_evaluation(self, policy_path=None, tolerance=1e-12):
        """ Policy evaluation algorithm """

        if policy_path != None:
            with open(policy_path) as file:
                for i, line in enumerate(file):
                    self.policy[i] = int(line)

        self.value_function = np.zeros(self.nums_states)
        while True:
            error = 0
            value_matrix = np.zeros(self.nums_states)

            for state in range(self.nums_states):
                optimal_action = self.policy[state]
                value_matrix[state] = np.sum(self.mdp_transitions[state, int(optimal_action), :] * (
                    self.mdp_rewards[state, int(optimal_action), :] + self.discount_factor * self.value_function))

            error = np.max(np.abs(self.value_function - value_matrix))
            self.value_function = value_matrix
            if (error < tolerance):
                break

    def print_output(self):
        for i in range(self.nums_states):
            print(str(self.value_function[i]) +
                  " " + str(self.policy[i]))


def mdp_path(string):
    if os.path.isfile(string):
        # print("MDP Path used %s" % string)
        return string
    else:
        raise NotADirectoryError(string)


def policy_path(string):
    if os.path.isfile(string):
        # print("Policy being evaluated %s" % string)
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp", type=mdp_path)
    parser.add_argument("--algorithm", type=str, default="vi")
    parser.add_argument("--policy", type=policy_path)

    args = parser.parse_args()

    mdp = MDPPlanning(args.mdp)

    # print(mdp.nums_actions, mdp.nums_states,
    #       mdp.mdp_rewards[299][5][301], mdp.mdp_transitions[0][0][300], mdp.end_states)

    if (args.policy == None):
        # Computing optimal policy
        if not (args.algorithm == "hpi" or args.algorithm == "vi" or args.algorithm == "lp"):
            print("Algorithm should be hpi, vi or lp")
            sys.exit(0)
        elif args.algorithm == "vi":
            mdp.value_iteration()
            mdp.print_output()

        elif args.algorithm == "hpi":
            mdp.howards_policy_iteration()
            mdp.print_output()

        else:
            mdp.linear_programming()
            mdp.print_output()
            # print("%s Algorithm used" % args.algorithm)

    else:
        pass
        # Evaluating the given policy
        mdp.policy_evaluation(args.policy)
        mdp.print_output()