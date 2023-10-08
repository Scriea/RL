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
        self.discount = args[6]     # Discount factor (γ)
        if pol_args:
            pass                    # Policy
    
    def VI(self, epsilon=1e-11) -> tuple:
        """
        Value Iteration,
        Approach - II
        """
        V = np.random.random_sample(self.n)     # Value Function Initialization, Vo
        while True:    
            temp = np.sum(self.T*(self.R + self.discount*V), axis=2)
            V_temp = np.max(temp,axis=1)
            P_temp = np.argmax(temp, axis=1)    # Optimal policy            
            for s in range(self.n):
                if s in self.E:
                    V_temp[s] = 0
                    V[s] = 0
            if np.linalg.norm(V_temp - V)<epsilon:
                return V_temp, P_temp
            V = V_temp

 
    # def VI(self, epsilon=1e-11):
        """
        Approach - I
        Uses less memory but was slow
        """
        # P = np.zeros((self.n), dtype= int)      # Policy Function, π
        # V_t = np.full((self.n), - np.inf, dtype= float)  
        # for s in range(self.n):
        #     for a in range(self.k):
                # temp=0
                # for s_dash in range(self.n):
                #     if s_dash in self.E:
                #         temp += self.T[s][a][s_dash]*(self.R[s][a][s_dash])
                #     else:
                #         temp += self.T[s][a][s_dash]*(self.R[s][a][s_dash] + self.discount * V[s_dash])
                # if temp > V_t[s]:
                #     P[s] = a
                #     V_t[s] = temp
            # for s in range(self.n):
            #     if s in self.E:
            #         V_t[s]=0
            # if np.linalg.norm(V_t- V) < epsilon:
            #     V = V_t
            #     break
            # V = V_t
        # return (V, P)

    def EvaluatePolicy(self, pol, epsilon=1e-11):
        """ 
        Policy Evaluation 
        Returns Policy Value Functions, V_π for a given policy π
        """    
        V = np.random.random_sample(self.n)
        P = np.array(read_pol(pol), dtype= int)
        while True: 
            V_t = np.zeros(self.n)               
            for s in range(self.n):
                if s in self.E:
                    V[s] = 0     

            for s in range(self.n):   
                for s_dash in range(self.n): 
                    V_t[s] += (self.R[s][P[s]][s_dash] + V[s_dash]*self.discount)*self.T[s][P[s]][s_dash]
                if s in self.E:
                    V_t[s]=0

            if np.linalg.norm(V_t - V) < epsilon:
                V = V_t
                break
            V = V_t
        return V, P
    
    def LP(self) -> tuple:
        """
        Linear Programming for MDP
        """
        V = np.zeros((self.n), dtype= float)
        P = np.zeros((self.n), dtype= int)
        prob = LpProblem("MDP", LpMinimize)
        decision_vars = [ LpVariable("V"+ str(i)) for i in range(self.n)]           # Decision Variables i.e. V1, V2, ...        
        objective = lpSum(decision_vars)                                            # Objective Function i.e  Σ(V)
        prob += objective

        # Adding Constraints
        for s in range(self.n):
            for a in range(self.k):
                prob += decision_vars[s] >= (lpSum( [self.T[s][a][s_dash]*self.R[s][a][s_dash] if s_dash in self.E else 
                                                  self.T[s][a][s_dash]*(self.R[s][a][s_dash] + self.discount*decision_vars[s_dash]) 
                                                    for s_dash in range(self.n)])) 
        optimization_res = prob.solve(PULP_CBC_CMD(msg=False))
        for i in range(self.n):
            V[i] = decision_vars[i].varValue
            if i in self.E:
                V[i] = 0
    
        P = np.argmax(np.sum(self.T * (self.R + self.discount * V), axis=2), axis=1)
        return V, P
    
    def HPI(self) -> tuple:
        "Howard's Policy Iterantion"
        pass    

    def solver(self, algo, pol=None):
        if pol:
            return self.EvaluatePolicy(pol)
        else:
            if algo.lower() == 'hpi':
                return self.HPI()
            elif algo.lower() =='lp':
                return self.LP()
            else:                                                                   # Default setting
                return self.VI()
        

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

def read_pol(path) -> list:
    policy = []
    with open(path) as f:
        action = f.readline()
        while action:
            policy.append(int(action))
            action = f.readline()
    return policy

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
    V, P = mdp.solver(algo=ALGO, pol=PATH_POL)
    for i in range(len(V)):
        print(f"{V[i]:.6f} {P[i]}")