import numpy as np

def check_occurrence(permutation):
    for i in range(len(permutation) - 1):
        if (permutation[i] == 2 and permutation[i + 1] == 5) or (permutation[i] == 1 and permutation[i + 1] == 3):
            return 1
    return 0

permutations = np.array([np.random.permutation(np.arange(1,6)) for i in range(1000)])
matrix = permutations
print(matrix)

desired_occurrences = sum([check_occurrence(column) for column in matrix])
probability = desired_occurrences / 1000
print(f"Approximate Probability: {probability}")
