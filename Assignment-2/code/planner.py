import argparse
import numpy as np
from pulp import *


## Const Vars
default_algorithm = "vi"
precision = 1e-8    # Value Iteration Precision


## Helper Classe
class MDP:
    def __init__(self, args) -> None:
        self.n = args[0]    # No. of States
        self.k = args[1]    # No. of Actions
        self.S = np.arange(self.n, dtype=int) 
        self.A = np.arange(self.k, dtype=int)
        self.E = args[2]    # List of end states
        self.algo


    def PE(self, pol)->None:
        """
        Policy Evaluation
        Returns Policy Value Functions 
        for a given policy
        """    
    def VI(self, epsilon=precision)-> None:
        """
        Value Iteration
        Returns Value Functions of MDP
        """
        pass
    def HPI(self) -> None:
        "Howard's Policy Iterantion"
        pass
    def LP(self) -> None:
        "Linear Programming"
        pass

    def solver(self, algo, pol=None) -> None:
        if pol:
            return self.PE(pol)
        else:
            if algo=="vi":
                return self.VI()
            elif algo=='hpi':
                return self.HPI()
            elif algo=='LP':
                return self.LP()
            else:
                print("Error! This algorithm isn't developed.")
                print("Please provide appropriate input")
                exit()

## Helper Functions
def read_MDP(path) -> list:
    args = []
    with open(path, 'r') as f:
        pass

    return args

## Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mdp",type=str, required= True, 
                        help="Path to mdp file")
    parser.add_argument("--algorithm",type=str,default=default_algorithm, 
                        help="Specify the algorithm to calculate Value Functions, (vi/hpi/lp)")
    parser.add_argument("--policy",type=str, default= None,
                         help="Path to policy file")

    args = parser.parse_args()
    PATH_MDP = args.mdp
    ALGO = args.algorithm
    PATH_POL = args.policy

    argv = read_MDP(PATH_MDP)

