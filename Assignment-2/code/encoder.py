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

def check_tackling(pos)-> bool:
    pass

def check_collinear(pos)-> bool:
    pos_P1 = posToCoor(pos[0:2])
    pos_P2 = posToCoor(pos[2:4])
    pos_R = posToCoor(pos[4:6])
    return (pos_P1[1]- pos_P2[1])/(pos_P1[0]-pos_P2[0])== (pos_P1[1]- pos_R[1])/(pos_P1[0]-pos_R[0])

def pass_prob(pos, q)-> float:
    pos_P1 = posToCoor(pos[0:2])
    pos_P2 = posToCoor(pos[2:4])
    pos_R = posToCoor(pos[4:6])

    if check_collinear(pos):
        return 0.5*(q- 0.1*max(abs(pos_P1[0]-pos_P2[0]), abs(pos_P1[1]-pos_P2[1])))
    else:
        return (q- 0.1*max(abs(pos_P1[0]-pos_P2[0]), abs(pos_P1[1]-pos_P2[1])))

def shooting_prob(pos, q)->float:
    pos_P1 = posToCoor(pos[0:2])
    pos_P2 = posToCoor(pos[2:4])
    pos_R = posToCoor(pos[4:6])

    x = pos_P1[0] if pos[6]=="1" else pos_P2[0]
    if int(pos[4:6]) in [8,12]:
        return 0.5*(q - 0.2*(3-x))
    else:
        return (q - 0.2*(3-x))

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
    print("end 0 8193")                 # End States, Winning State, Losing State

    #Transitions
    T = np.zeros((8194, 10, 8194))
    R = np.zeros((8194, 10, 8194))

    for s in range(1, 8193):
        opp_move_prob = opp_move[s-1]
        for a in range(10):
            pos = stateTopos[s]

            """
            Moving Player 1
            """ 
            if a in [0,1,2,3]:  
                pos = numTopos(int(pos[:2])+movePosnum[a]) + pos[2:]
                if pos not in posTostate.keys():
                    T[s][a][0] = 1
                    R[s][a][0] = -1
                    print(f"transition {s} {a} 0 {R[s][a][0]} {T[s][a][0]}")

                else:
                    #Player moving with Ball                    
                    if pos[6]=="1":         
                        for i, prob in enumerate(opp_move_prob):
                            new_pos = pos[:4] + numTopos(int(pos[4:6]) + movePosnum[i]) + pos[6:]
                            if new_pos not in posTostate.keys():
                                continue

                            if check_tackling(pos, new_pos):                    # Tackling Condition
                                pass
                            else:                
                                s_dash = posTostate[new_pos]
                                T[s][a][s_dash] = (1-2*p)*prob                  # Succesfull movement
                                print(f"transition {s} {a} {s_dash} {R[s][a][s_dash]} {T[s][a][s_dash]}")
                        T[s][a][0] = 2*p                                        # Losing the ball directly
                        T[s][a][0] = -1
                        print(f"transition {s} {a} 0 {R[s][a][0]} {T[s][a][0]}")

                    # Player moving without ball
                    else:
                        for i, prob in enumerate(opp_move_prob):
                            new_pos = pos[:4] + numTopos(int(pos[4:6]) + movePosnum[i]) + pos[6:]
                            if not new_pos in posTostate.keys():                # Episode Ends, Player has gone out of the field
                                continue
                       
                            s_dash = posTostate[new_pos]
                            T[s][a][s_dash] = (1-p)*prob                        # Succesfull movement
                            print(f"transition {s} {a} {s_dash} {R[s][a][s_dash]} {T[s][a][s_dash]}")
                        T[s][a][0] = p                                          # Losing the ball directly
                        print(f"transition {s} {a} 0 {R[s][a][0]} {T[s][a][0]}")
                """
                Motion Player 2
                """     
            elif a in [4,5,6,7]:                            
                pos = pos[:2] + numTopos(int(pos[2:4])+movePosnum[a-4]) + pos[4:]
                if pos not in posTostate.keys():
                    T[s][a][0] = 1
                    R[s][a][0] = -1
                    print(f"transition {s} {a} 0 {R[s][a][0]} {T[s][a][0]}")

                else:
                    #Player moving with Ball                    
                    if pos[6]=="2":         
                        for i, prob in enumerate(opp_move_prob):
                            new_pos = pos[:4] + numTopos(int(pos[4:6]) + movePosnum[i]) + pos[6:]
                            if new_pos not in posTostate.keys():
                                continue

                            if check_tackling(pos, new_pos):                    # Tackling Condition
                                pass
                            else:                
                                s_dash = posTostate[new_pos]
                                T[s][a][s_dash] = (1-2*p)*prob                  # Succesfull movement
                                print(f"transition {s} {a} {s_dash} {R[s][a][s_dash]} {T[s][a][s_dash]}")
                        T[s][a][0] = 2*p                                        # Losing the ball directly
                        T[s][a][0] = -1
                        print(f"transition {s} {a} 0 {R[s][a][0]} {T[s][a][0]}")

                    # Player moving without ball
                    else:
                        for i, prob in enumerate(opp_move_prob):
                            new_pos = pos[:4] + numTopos(int(pos[4:6]) + movePosnum[i]) + pos[6:]
                            if not new_pos in posTostate.keys():                # Episode Ends, Player has gone out of the field
                                continue
                       
                            s_dash = posTostate[new_pos]
                            T[s][a][s_dash] = (1-p)*prob                        # Succesfull movement
                            print(f"transition {s} {a} {s_dash} {R[s][a][s_dash]} {T[s][a][s_dash]}")
                        T[s][a][0] = p                                          # Losing the ball directly
                        print(f"transition {s} {a} 0 {R[s][a][0]} {T[s][a][0]}")

                """
                Passing
                """    
            elif a==8:    
                pos = stateTopos[s]         
                pos = pos[:6] + str(3-int(pos[6]))                   
                for i, prob in enumerate(opp_move_prob):
                    new_pos = pos[:4] + numTopos(int(pos[4:6]) + movePosnum[i]) + pos[6:] 
                    if new_pos not in posTostate.keys():
                        continue
                    s_dash = posTostate[new_pos]    
                    p_prob =  pass_prob(new_pos, q)
                    T[s][a][s_dash] = p_prob*prob
                    print(f"transition {s} {a} {s_dash} {R[s][a][s_dash]} {T[s][a][s_dash]}")
                    T[s][a][0] += (1-prob)*prob
                print(f"transition {s} {a} 0 {R[s][a][0]} {T[s][a][0]}")
            

                """
                Shooting
                """
            elif a==9:
                pos = stateTopos[s]
                print(f"transition {s} {a} {s_dash} {R[s][a][s_dash]} {T[s][a][s_dash]}")

    print('mdptype episodic')
    print('discount 1.0')
