import numpy as np

prob_x = [0.1, 0.2, 0.2, 0.3, 0.1, 0.1]
prob_y = [0.1, 0.3, 0.1, 0.1, 0.3, 0.1]
prob_z = [0.2, 0.1, 0.1, 0.2, 0.1, 0.3]

num_simulations = 100000

X = np.random.choice(np.arange(1, 7), size=num_simulations, p=prob_x)
Y = np.random.choice(np.arange(1, 7), size=num_simulations, p=prob_y)
Z = np.random.choice(np.arange(1, 7), size=num_simulations, p=prob_z)

Q = X**3 + Y**3 + Z**3

prob_q_between_200_and_300 = np.sum((Q >= 200) & (Q <= 300)) / num_simulations

print("Probability that Q lies between 200 and 300:", prob_q_between_200_and_300)