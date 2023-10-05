import argparse
import numpy as np
from pulp import *
np.random.seed(37)


## Helper Classe
class MDP:
    def __init__(self, args, pol_args=None) -> None:
        self.n = args[0]            # No. of States
        self.k = args[1]            # No. of Actions
        self.E = args[2]            # List of end states
        self.T = args[3]            # Transisition Function/Probabilty
        self.R = args[4]            # Reward Function
        self.type = args[5]         # MDP Type (continuing/episodic)
        self.discount = args[6]     # MDP Discount factor(gamma)
        if pol_args:
            pass                    # Policy

    def EvaluatePolicy(self, pol)->None:
        """ 
        Policy Evaluation 
        Returns Policy Value Functions, V_p for a given policy p
        """    


    def VI(self, epsilon=1e-7):
        "Value Iteration"
        P = np.ones((self.n))                   # Policy
        V = np.random.random_sample(self.n)     # Value Function Initialization, Vo
        while True:    
            B_V = self.BOO(V)
            if np.linalg.norm(B_V - V) < epsilon:
                break
            V = B_V
        return B_V, P
    
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

    def BOO(self, V):
        """
        Bellman Optimality Operator (BOO)
        Returns B*(V)
        """
        p = np.zeros((self.n), dtype= int)              # Optimal Policy
        V_t = np.full((self.n), -np.inf, dtype= float)  # 
        for s in range(self.n):
            for a in range(self.k):
                for s_dash in range(self.n):
                    temp = 0
                    if s_dash in self.E:
                        temp+= self.T[s][a][s_dash]*(self.R[s][a][s_dash])
                    else:
                        temp += self.T[s][a][s_dash]*(self.R[s][a][s_dash] + self.discount * V[s_dash])
                    V_t[s] = max(temp, V_t[s])
                    if temp > V_t[s]:
                        p[s] = a
                        V_t[s] = temp

        for s in range(self.n):
            if s in self.E:
                V_t[s] = 0
        return V_t, p

## Helper Functions

def read_MDP(path) -> list:
    args = []
    with open(path, 'r') as f:
        n = int(f.readline().split()[-1])               # No of States       
        k = int(f.readline().split()[-1])               # No of Actions
        e = list(map(int,f.readline().split()[1:]))     # List of End States
        T = np.zeros((n, k, n))                         # Transition Function/Probability
        R = np.zeros((n, k, n))                         # Reward Function
        temp = f.readline().split()
        while temp[0]=="transition":
            vals = list(map(int,temp[1:4])) + list(map(float, temp[4:]))
            R[vals[0]][vals[1]][vals[2]] = vals[3]
            T[vals[0]][vals[1]][vals[2]] = vals[4]
            temp = f.readline().split()
        mdptype = temp[1]
        discount = float(f.readline().split()[-1])
    f.close()
    return [n, k, e, T, R, mdptype, discount]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mdp",type=str, required= True, 
                        help="Path to mdp file")
    parser.add_argument("--algorithm",type=str,default='vi',              # default algorithm is Value iteration
                        help="Specify the algorithm to calculate Value Functions, (vi/hpi/lp)")
    parser.add_argument("--policy",type=str, default= None,
                         help="Path to policy file")
    
    args = parser.parse_args()
    PATH_MDP = args.mdp
    ALGO = args.algorithm
    PATH_POL = args.policy

    mdp_args = read_MDP(PATH_MDP)
    mdp = MDP(args=mdp_args)
    # print(mdp_args)



