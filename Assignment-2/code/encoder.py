import argparse
import numpy as np

## Helper Functions & Vars
def numTopos(num):
    if num<10:
        return "0"+str(num)
    else:
        return str(num)

def inField(pos,a):
    if a==0 and int(pos[0:2]) in [1,5,9,13]:
        return False
    if a==1 and int(pos[0:2]) in [4,8,12,16]:
        return False
    if a==2 and int(pos[0:2]) in [1,2,3,4]:
        return False
    if a==3 and int(pos[0:2]) in [13,14,15,16]:
        return False
    if a==4 and int(pos[2:4]) in [1,5,9,13]:
        return False
    if a==5 and int(pos[2:4]) in [4,8,12,16]:
        return False
    if a==6 and int(pos[2:4]) in [1,2,3,4]:
        return False
    if a==7 and int(pos[2:4]) in [13,14,15,16]:
        return False

    return True

def DefernderInField(pos, a):
    if a==0 and int(pos[4:6]) in [1,5,9,13]:
        return False
    if a==1 and int(pos[4:6]) in [4,8,12,16]:
        return False
    if a==2 and int(pos[4:6]) in [1,2,3,4]:
        return False
    if a==3 and int(pos[4:6]) in [13,14,15,16]:
        return False

    return True

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

"""
Returns Distance Square
"""
def dist(p_a, p_b):
    return np.sqrt((p_a[0] -p_b[0])**2 + (p_a[1] - p_b[1])**2)

def check_tackling(pos, new_pos)-> bool:
    pos_P1_i = int(pos[0:2])
    pos_P2_i = int(pos[2:4])
    pos_R_i = int(pos[4:6])
    pos_P1_f = int(new_pos[0:2])        
    pos_P2_f = int(new_pos[2:4])
    pos_R_f = int(new_pos[4:6])
    if new_pos[6]=="1":
        if pos_P1_f == pos_R_f:                                # Check - share same square
            return True
        if pos_P1_i == pos_R_f and pos_P1_f==pos_R_i:                 # Check - Cross
            return True
    else:
        if pos_P2_f == pos_R_f:
            return True
        if pos_P2_i == pos_R_f and pos_P2_f==pos_R_i:
            return True
        
    return False
    
def check_collinear(pos)-> bool:

    """
    Simplest way of checking both collinearity and
    that defender lies between players is using
    distances among them.
    """
    pos_P1 = posToCoor[pos[0:2]]
    pos_P2 = posToCoor[pos[2:4]]
    pos_R = posToCoor[pos[4:6]]

    return (dist(pos_P1, pos_R) + dist(pos_P2, pos_R)- dist(pos_P1, pos_P2)) < 1e-5 

def pass_prob(pos, q)-> float:
    pos_P1 = posToCoor[pos[0:2]]
    pos_P2 = posToCoor[pos[2:4]]

    if check_collinear(pos):
        return (q- 0.1*max(abs(pos_P1[0]-pos_P2[0]), abs(pos_P1[1]-pos_P2[1])))/2
    else:
        return (q- 0.1*max(abs(pos_P1[0]-pos_P2[0]), abs(pos_P1[1]-pos_P2[1])))

def shooting_prob(pos, q)->float:
    pos_P1 = posToCoor[pos[0:2]]
    pos_P2 = posToCoor[pos[2:4]]

    x = pos_P1[0] if pos[6]=="1" else pos_P2[0]
    if int(pos[4:6]) in [8,12]:
        return (q - 0.2*(3-x))/2
    else:
        return (q - 0.2*(3-x))
    
# def get_pos(pos):
#     if int(pos[0:2]) > int(pos[2:4]):
#         return pos[2:4] + pos[0:2] + pos[4:]
#     else:
#         return pos
    
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
    return stateTopos, posTostate, opp_move

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
    print("numStates 8194")                     # Game Positions(8192) + Win state(1) + Lose state(1)
    print("numActions 10")
    print("end 0 8193")                         # End States, Winning State, Losing State

    # Transitions
    T = np.zeros((8194, 10, 8194), dtype= float)

    for s in range(1, 8193):
        opp_move_prob = opp_move[s-1]
        for a in range(10):
            if a in [0,1,2,3]:  
                """
                Moving Player 1
                """ 
                pos = stateTopos[s]
                if not inField(pos, a):                                         # Check if Player moves out of field 
                    T[s][a][0] = 1
                    print(f"transition {s} {a} 0 0 {T[s][a][0]}")
                
                else:
                    p_pos = numTopos(int(pos[:2]) +movePosnum[a]) + pos[2:]   
                    #Player moving with Ball                    
                    if pos[6]=="1":         
                        for i, prob in enumerate(opp_move_prob):
                            if not DefernderInField(pos, i): 
                                continue
                            new_pos = p_pos[:4] + numTopos(int(p_pos[4:6]) + movePosnum[i]) + p_pos[6:]
                            s_dash = posTostate[new_pos]

                            # Tackling Occurs
                            if check_tackling(pos, new_pos):                    
                                T[s][a][s_dash] = (0.5-p)*prob                  # Player wins Tackle
                                print(f"transition {s} {a} {s_dash} 0 {T[s][a][s_dash]}")
                                T[s][a][0] += (0.5+p)*prob                      # Player Loses Tackle   
                            
                            # Tackling Doesn't Occur, Movement only
                            else:                
                                T[s][a][s_dash] = (1-2*p)*prob                  # Succesfull movement
                                print(f"transition {s} {a} {s_dash} 0 {T[s][a][s_dash]}")
                                T[s][a][0] += 2*p*prob                          # Losing the ball directly
                        print(f"transition {s} {a} 0 0 {T[s][a][0]}")

                    # Player moving without ball
                    else:
                        for i, prob in enumerate(opp_move_prob):
                            if not DefernderInField(pos, i):
                                continue
                            new_pos = p_pos[:4] + numTopos(int(p_pos[4:6]) + movePosnum[i]) + p_pos[6:]
                            s_dash = posTostate[new_pos]
                            T[s][a][s_dash] = (1-p)*prob                        # Succesfull movement
                            print(f"transition {s} {a} {s_dash} 0 {T[s][a][s_dash]}")
                        T[s][a][0] = p                                          # Losing the ball directly
                        print(f"transition {s} {a} 0 0 {T[s][a][0]}")

            elif a in [4,5,6,7]: 
                """
                Motion Player 2
                """    
                pos = stateTopos[s] 
                if not inField(pos, a):
                    T[s][a][0] = 1
                    print(f"transition {s} {a} 0 0 {T[s][a][0]}")      
                else:
                    p_pos = pos[:2] + numTopos(int(pos[2:4])+ movePosnum[a-4]) + pos[4:]
                    #Player moving with Ball                    
                    if pos[6]=="2":       
                        for i, prob in enumerate(opp_move_prob):
                            if not DefernderInField(pos, i):
                                continue
                            new_pos = p_pos[:4] + numTopos(int(p_pos[4:6]) + movePosnum[i]) + p_pos[6:]
                            s_dash = posTostate[new_pos]
                            if check_tackling(pos, new_pos):                    # Tackling Condition
                                T[s][a][s_dash] = (0.5-p)*prob
                                print(f"transition {s} {a} {s_dash} 0 {T[s][a][s_dash]}")
                                T[s][a][0] += (0.5+p)*prob
                            else:                
                                T[s][a][s_dash] = (1-2*p)*prob                  # Succesfull movement
                                print(f"transition {s} {a} {s_dash} 0 {T[s][a][s_dash]}")
                                T[s][a][0] += 2*p*prob                          # Losing the ball directly
                        print(f"transition {s} {a} 0 0 {T[s][a][0]}")

                    # Player moving without ball
                    else:
                        for i, prob in enumerate(opp_move_prob):
                            if not DefernderInField(pos, i):
                                continue
                            new_pos = p_pos[:4] + numTopos(int(p_pos[4:6]) + movePosnum[i]) + p_pos[6:]
                            s_dash = posTostate[new_pos]
                            T[s][a][s_dash] = (1-p)*prob                        # Succesfull movement
                            print(f"transition {s} {a} {s_dash} 0 {T[s][a][s_dash]}")
                        T[s][a][0] = p                                          # Losing the ball directly
                        print(f"transition {s} {a} 0 0 {T[s][a][0]}")

                """
                Passing
                """    
            elif a==8:    
                pos = stateTopos[s]         
                pos = pos[:6] + str(3-int(pos[6]))                   
                for i, prob in enumerate(opp_move_prob):
                    if not DefernderInField(pos, i):
                        continue
                    new_pos = pos[:4] + numTopos(int(pos[4:6]) + movePosnum[i]) + pos[6] 
                    s_dash = posTostate[new_pos]    
                    s_prob =  pass_prob(new_pos, q)
                    T[s][a][s_dash] = s_prob*prob
                    print(f"transition {s} {a} {s_dash} 0 {T[s][a][s_dash]}")
                    T[s][a][0] += (1.0-s_prob)*prob
                print(f"transition {s} {a} 0 0 {T[s][a][0]}")

                """
                Shooting
                """
            else:
                pos = stateTopos[s]
                for i, prob in enumerate(opp_move_prob):
                    if not DefernderInField(pos, i):
                        continue
                    new_pos = pos[:4] + numTopos(int(pos[4:6]) + movePosnum[i]) + pos[6:] 
                    p_prob =  shooting_prob(new_pos, q)
                    T[s][a][8193] += p_prob*prob
                    T[s][a][0] += (1.0-p_prob)*prob

                print(f"transition {s} {a} 8193 1 {T[s][a][8193]}")    
                print(f"transition {s} {a} 0 0 {T[s][a][0]}")

    print('mdptype episodic')
    print('discount 1.0')