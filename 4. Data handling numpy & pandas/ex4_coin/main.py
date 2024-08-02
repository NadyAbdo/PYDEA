import numpy as np
import math
import matplotlib.pyplot as plt

def calculate_probability(n, p, condition):
    probabilities = [math.comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in range(n + 1) if condition(k)]
    return np.sum(probabilities)

def play_game(principal, num_games):
    cond1, cond2 = 44, 30
    principal_values = [principal]

    for _ in range(num_games):
        # Calculate probabilities for the current conditions
        prob_fair = calculate_probability(100, 0.5, lambda k: k >= cond1)
        prob_unfair = calculate_probability(100, 0.3, lambda k: k <= cond2)

        # Probability of winning the bet (AND condition)
        prob_bet_won = prob_fair * prob_unfair

        # Simulate a win or loss
        if np.random.rand() < prob_bet_won:
            cond1 += 1
            cond2 -= 1
            principal += 1000
        else:
            cond1 -= 1
            cond2 += 1
            principal -= 1000

        # Save the updated principal value
        principal_values.append(principal)
    return principal_values

n_fair = 100
p_fair = 0.5

n_unfair = 100
p_unfair = 0.3

condition_fair = lambda k: k >= 44
condition_unfair = lambda k: k <= 30

prob_fair = calculate_probability(n_fair, p_fair, condition_fair)
prob_unfair = calculate_probability(n_unfair, p_unfair, condition_unfair)

prob_bet_won = prob_fair * prob_unfair

print(f"Probability of winning the bet: {prob_bet_won:.4f}")

initial_principal = 20000
num_simulations = 1000000

result = play_game(initial_principal, num_simulations)

plt.plot(result)
plt.xlabel('Number of Games')
plt.ylabel('PRINCIPAL')
plt.title('Evolution of PRINCIPAL vs. Number of Games')
plt.show()

'''
import numpy as np
import matplotlib.pyplot as plt

def calculate_probability(n, p, no):
    probabilities = np.random.binomial(n, p, size=no)
    return probabilities

def play_game(principal, num_games):
    cond1, cond2 = 44, 30
    principal_values = [principal]
    prob = 0

    for _ in range(num_games):
        # Calculate probabilities for the current conditions
        prob_fair = calculate_probability(n_fair, p_fair, 1)
        prob_unfair = calculate_probability(n_unfair, p_unfair, 1)

        # Probability of winning the bet (AND condition)
        for i in range(len(prob_fair)) :
            if (prob_fair>cond1-1 and prob_unfair<cond2+1) :
                prob_bet_won = 'win'

            else : 
                prob_bet_won = 'lose'

            # Simulate a win or loss
            if prob_bet_won == 'win':
                cond1 += 1
                cond2 -= 1
                principal += 1000
            else:
                cond1 -= 1
                cond2 += 1
                principal -= 1000

        # Save the updated principal value
        principal_values.append(principal)
    return principal_values

n_fair = 100
p_fair = 0.5

n_unfair = 100
p_unfair = 0.3

prob = 0

prob_fair = calculate_probability(n_fair, p_fair, 1000000)
prob_unfair = calculate_probability(n_unfair, p_unfair, 1000000)

for i in range(len(prob_fair)) :
    if (prob_fair[i]>43 and prob_unfair[i]<31) :
        prob = prob + 1

print(f"Probability of winning the bet: {(prob)/1000000:.4f}")

initial_principal = 20000
num_simulations = 1000000

result = play_game(initial_principal, num_simulations)

plt.plot(result)
plt.xlabel('Number of Games')
plt.ylabel('PRINCIPAL')
plt.title('Evolution of PRINCIPAL vs. Number of Games')
plt.show()
'''