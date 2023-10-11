import argparse

## Helper Functions & Vars
S_coor= { 1: (0,0),
          2: (1,0),
          3: (2,0),
          4: (3,0),
          5: (0,1),
          6: (1,1),
          7: (2,1),
          8: (3,1),
          9: (0,2),
          10:(1,2),
          11:(2,2),
          12:(3,2),
          13:(0,3),
          14:(1,3),
          15:(2,3),
          16:(3,3)
          }

def read_opponent(path):
    with open(path) as f:
        f.readline()
        temp = f.readline()
        while temp:
            
            temp = f.readline()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", required= True, type= str, help="Path to opponents Policy")
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--q", type=float, required=True)
    args = parser.parse_args()
    PATH_OPP = args.opponent

    print("numStates 8193")
    print("numActions 10")
    ## Endstates
    print("end 0")
    ##Transitions
    for s in range(1,8193):
        for a in range(10):
            T = None
            R = None
            s_dash = None

            print(f"transition {s} {a} {s_dash} {R} {T}")


    print('mdptype episodic')
    print('discount 1.0')
