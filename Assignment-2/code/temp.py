import numpy as np

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



stateTopos, posTostate, opp_move = read_opponent("./data/football/test-1.txt")

print(posTostate["0102041"])


# def check_collinear(pos) -> bool:
#     pos_P1 = posToCoor[pos[0:2]]
#     pos_P2 = posToCoor[pos[2:4]]
#     pos_R = posToCoor[pos[4:6]]
    
#     if abs(dist(pos_P1, pos_R) + dist(pos_P2, pos_R)- dist(pos_P1, pos_P2)) < 1e-3:             # Defender Lies between
#         if pos_P1[0] == pos_P2[0]:
#             return True
#         elif pos_P1[1] == pos_P2[1]:
#             return True
#         elif abs(pos_P1[1] - pos_P2[1]) == abs(pos_P1[0] - pos_P2[0]):
#             return True
#         else:
#             return False
#     return False