import numpy as np

def evaluate_fitness(values, weights, capacity, selection):
    """
    Calculate the fitness of a single knapsack solution.

    Args:
        values (np.ndarray): Array of item values.
        weights (np.ndarray): Array of item weights.
        capacity (float): Knapsack capacity.
        selection (np.ndarray): Binary selection array (0 or 1).

    Returns:
        float: Fitness value of the solution.
    """
    total_value = np.dot(values, selection)
    total_weight = np.dot(weights, selection)
    if total_weight <= capacity:
        return total_value
    return 0  # Penalize infeasible solutions


def evaluate_fitness_batch(values, weights, capacity, selections):
    """
    Calculate fitness for a batch of knapsack solutions.

    Args:
        values (np.ndarray): Array of item values.
        weights (np.ndarray): Array of item weights.
        capacity (float): Knapsack capacity.
        selections (np.ndarray): 2D array of binary selection vectors.

    Returns:
        np.ndarray: Array of fitness values for the batch.
    """
    fitnesses = []
    for selection in selections:
        fitnesses.append(evaluate_fitness(values, weights, capacity, selection))
    return np.array(fitnesses)

# Test instance data

# values = np.array([10, 20, 30, 40, 50])  # Item values
# weights = np.array([1, 2, 3, 8, 7])      # Item weights
# capacity = 10                            # Knapsack capacity
# binary_matrix = np.array([               # Binary selection vectors (batch)
#     [1, 0, 1, 0, 1],  # Select items 1, 3, 5
#     [0, 1, 0, 1, 0],  # Select items 2, 4
#     [1, 1, 1, 0, 0],  # Select items 1, 2, 3
#     [0, 0, 0, 0, 0]   # Select no items
# ])

# # Calculate fitness for the batch
# fitness_values = evaluate_fitness_batch(values, weights, capacity, binary_matrix)

# # Print results
# print("Knapsack Test Data:")
# print("Values:", values)
# print("Weights:", weights)
# print("Capacity:", capacity)
# print("\nBinary Selection Matrix (Batch):")
# print(binary_matrix)
# print("\nFitness Values:")
# print(fitness_values)