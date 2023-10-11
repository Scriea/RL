import numpy as np
import argparse

## Helper Functions

def read_opponent(path):
    stateTopos = {0:"0", 8193:"1"}                               
    posTostate = {"0000000":0, "1000000": 8193}
    opp_move = []
    i= 1
    with open(path) as f:
        f.readline()
        temp = f.readline()
        while temp:
            te = temp.split()
            stateTopos[i] = te[0]
            posTostate[te[0]] = i
            opp_move.append([ float(i) for i in te[1:]])
            temp = f.readline()
            i+=1
    return stateTopos, posTostate, np.array(opp_move)

def read_value(path):
    V = []
    P = []
    with open(path) as f:
        temp = f.readline()
        while temp:
            V.append(float(temp.split()[0]))
            P.append(int(temp.split()[1]))
            temp = f.readline()
        return V, P

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", required= True, type= str, help="Path to opponents Policy")
    parser.add_argument("--value-policy", required= True, type=str, help="Path to optimal policy and value")
    args = parser.parse_args()

    PATH_VALUE = args.value_policy
    PATH_OPPONENET = args.opponent
    V,P = read_value(PATH_VALUE)
    S, _, _ = read_opponent(PATH_OPPONENET)

    for i in range(1,8193):
        print(S[i], P[i], V[i])



