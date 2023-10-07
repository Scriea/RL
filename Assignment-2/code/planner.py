import argparse
import numpy as np
from pulp import *
np.random.seed(37)


## Helper Classe
class MDP:
    def __init__(self, args, pol_args=None):
        self.n = args[0]            # No. of States
        self.k = args[1]            # No. of Actions
        self.E = args[2]            # List of end states
        self.T = args[3]            # Transisition Function/Probabilty
        self.R = args[4]            # Reward Function
        self.type = args[5]         # MDP Type (continuing/episodic)
        self.discount = args[6]     # MDP Discount factor(gamma)
        if pol_args:
            pass                    # Policy

    def VI(self, epsilon=1e-11) -> tuple:
        """
        Value Iteration
        """
        # P = np.zeros((self.n), dtype= int)      # Policy Function, π
        V = np.random.random_sample(self.n)     # Value Function Initialization, Vo
        # t = 0
        while True:    
            # t+=1
            temp = np.sum(self.T*(self.R + self.discount*V), axis=2)
            V_temp = np.max(temp,axis=1)
            P_temp = np.argmax(temp, axis=1)
            for s in range(self.n):
                if s in self.E:
                    V_temp[s] = 0
                    V[s] = 0
            if np.linalg.norm(V_temp - V)<epsilon:
                # print(t)
                return V_temp, P_temp
            V = V_temp

            """
            This code was working but was vary slow,I used 
            numpy functions for calculations to improve speed.
            """
            # V_t = np.full((self.n), 2e-308, dtype= float)  
            # for s in range(self.n):
            #     for a in range(self.k):
                    # temp=0
                    # for s_dash in range(self.n):
                    #     if s_dash in self.E:
                    #         temp += self.T[s][a][s_dash]*(self.R[s][a][s_dash])
                    #     else:
                    #         temp += self.T[s][a][s_dash]*(self.R[s][a][s_dash] + self.discount * V[s_dash])
                    #     #V_t[s] = max(temp, V_t[s])
                    # if temp > V_t[s]:
                    #     P[s] = a
                    #     V_t[s] = temp
            # for s in range(self.n):
            #     if s in self.E:
            #         V_t[s]=0

            # print(V_t)     
            # print(V,P)
            # if np.linalg.norm(V_t- V) < epsilon:
            #     V = V_t
            #     break
            # V = V_t
        # print(t)
        # return (V, P)
    
    def LP(self) -> tuple:
        """
        Linear Programming
        """
        prob = LpProblem("MDP", LpMaximize)
        decision_vars = [ LpVariable("V"+str(i)) for i in range(self.n)]            # Decision Variables
        
        objective = lpSum(decision_vars)                                            # Objective Function
        prob += objective

        constraints = []
        for s in range(self.n):
            for a in range(self.k):
                for s_dash in range(self.n):
                    pass

        optimization_result = prob.solve()

        assert optimization_result == LpStatusOptimal
        print("Status:", LpStatus[prob.status])
        print("Optimal Solution to the problem: ", value(prob.objective))
        print ("Individual decision_variables: ")
        for v in prob.variables():
            print(v.name, "=", v.varValue)
        return (self.n)
    
    def HPI(self) -> tuple:
        "Howard's Policy Iterantion"
        pass    

    def solver(self, algo, pol=None):
        if pol:
            return self.PE(pol)
        else:
            if algo.lower() == 'hpi':
                return self.HPI()
            elif algo.lower() =='lp':
                return self.LP()
            else:                       ## Default setting
                return self.VI()
    def EvaluatePolicy(self, pol):
        """ 
        Policy Evaluation 
        Returns Policy Value Functions, V_π for a given policy π
        """    
        pass

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
    V, P = mdp.solver(algo=ALGO)
    for i in range(len(V)):
        print(f"{V[i]:.6f} {P[i]}")