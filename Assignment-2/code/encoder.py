import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--opponent", required= True, type= str,
                    help="Path to opponent")
parser.add_argument("--p", type=float, required=True,
                    help="p")
parser.add_argument("--q", type=float, required=True,
                    help="q")

args = parser.parse_args()
PATH_OPP = args.opponent


print("numStates 8192")
print("numActions 10")

## Endstates

##Transitions


print('mdptype episodic')
print('discount 1.0')
