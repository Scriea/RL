import random

def choose_direction(probabilities):
    # Generate a random number between 0 and 1
    rand_num = random.random()
    
    # Initialize the cumulative probability
    cumulative_prob = 0
    
    # Iterate through the probabilities and choose a direction
    for i, prob in enumerate(probabilities):
        cumulative_prob += prob
        if rand_num <= cumulative_prob:
            if i == 0:
                return "Left"
            elif i == 1:
                return "Right"
            elif i == 2:
                return "Up"
            else:
                return "Down"

# Example probabilities
Prob = [0.3, 0.3, 0.3, 0.1]

# Get the chosen direction
chosen_direction = choose_direction(Prob)

output = {"Left":0,
          "Right":0,
          "Up":0,
          "Down": 0}

for i in range(1000000):
    output[choose_direction(Prob)] +=1


print(output)
