import argparse
import numpy as np

## Helper Functions & Vars
def numTopos(num):
    if num<10:
        return "0"+str(num)
    else:
        return str(num)

movePosnum = [-1, 1, -4, +4]
    
posToCoor= {"01": (0,0),
            "02": (1,0),
            "03": (2,0),
            "04": (3,0),
            "05": (0,1),
            "06": (1,1),
            "07": (2,1),
            "08": (3,1),
            "09": (0,2),
            "10": (1,2),
            "11": (2,2),
            "12": (3,2),
            "13": (0,3),
            "14": (1,3),
            "15": (2,3),
            "16": (3,3)
            }

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

def choose_direction(movement_prob):
    rand_num = np.random.random()           # Generate a random number in [0,1]
    cumulative_prob = 0
    for i, prob in enumerate(movement_prob):
        cumulative_prob += prob
        if rand_num <= cumulative_prob:
            return i

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", required= True, type= str, help="Path to opponents Policy")
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--q", type=float, required=True)
    args = parser.parse_args()
    PATH_OPP = args.opponent
    p = args.p
    q = args.q
    stateTopos, posTostate, opp_move = read_opponent(PATH_OPP)
    
    """
    Encoding
    """

    print("numStates 8194")             # Game Positions(8192) + Win state(1) + Lose state(1)
    print("numActions 10")
    ## Endstates
    print("end 0 8193")

    #Transitions
    T = np.zeros((8194, 10, 8194))
    R = np.zeros((8194, 10, 8194))

    for s in range(1, 8193):
        pos = stateTopos[s]
        opp_move_prob = opp_move[s-1]
        for a in range(10):
            # Moving Player 1
            if a in [0,1,2,3]:  
                # Player has ball - tackling possible                        
                if pos[6]=="1":                         
                        if False:                                          # Trackling Condition
                            pass
                        else:
                            pos = numTopos(int(pos[:2])-movePosnum[a]) + pos[2:]
                            for i, prob in enumerate(opp_move_prob):
                                pos = pos[:4] + numTopos(int(pos[4:6])- movePosnum[i]) + pos[6:]
                                if not pos in posTostate.keys():               # Episode Ends, Player has gone out of the field
                                    s_dash = 0  
                                else:                
                                    s_dash = posTostate[pos]
                                T[s][a][s_dash] = (1-2*p)*prob                 # Succesfull movement
                                T[s][a][0] = 2*p*prob                          # Losing the ball directly
                
                # Player without ball
                else:
                    pos = numTopos(int(pos[:2])-movePosnum[a]) + pos[2:]
                    for i, prob in enumerate(opp_move_prob):
                        pos = pos[:4] + numTopos(int(pos[4:6])- movePosnum[i]) + pos[6:]
                        if not pos in posTostate.keys():        # Episode Ends, Player has gone out of the field
                            s_dash = 0  
                        else:                
                            s_dash = posTostate[pos]
                        T[s][a][s_dash] = (1-p)*prob                 # Succesfull movement
                        T[s][a][0] = p*prob                          # Losing the ball directly
            
            # Moving Player 2       
            elif a in [4,5,6,7]:                            
                if pos[6]=="2":                             # Player has ball,
                        pass  
                else:
                    pos = pos[0:2] + numTopos(int(pos[2:4])-movePosnum[a-4]) + pos[4:]
                    if not pos in posTostate.keys():        # Episode Ends, Player has gone out of the field
                        s_dash = 0  
                    else:                
                        s_dash = posTostate[pos]
                    T[s][a][s_dash] = 1-p
                    T[s][a][0] = p                           # Losing the ball directly
            
            # Passing Ball
            elif a==8:                                       
                pass
            # Shooting
            elif a==9:
                pass

    for s in range(8194):
        for a in range(10):
            for s_dash in range(8194):
                print(f"transition {s} {a} {s_dash} {R[s][a][s_dash]} {T[s][a][s_dash]}")

    print('mdptype episodic')
    print('discount 1.0')
