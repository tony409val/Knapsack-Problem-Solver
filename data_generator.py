import pulp
import random
from utils import save_data

# Generate knapsack instances with increasing difficulty in batches of 100
def generate_knapsack_instances(instance_type, num_items, instance_index, R=1000, H=100):
    items = []
    total_weight = 0

    # Generate items based on instance type
    for _ in range(num_items):
        if instance_type == 'UC':
            # Uncorrelated: Values and weights uniformly distributed over [1, R]
            value = random.randint(1, R)
            weight = random.randint(1, R)
        elif instance_type == 'SC':
            # Strongly Correlated: value = weight + R/10
            weight = random.randint(1, R)
            value = weight + R // 10
        elif instance_type == 'SS':
            # Subset Sum: value equals weight
            weight = random.randint(1, R)
            value = weight

        items.append((value, weight))
        total_weight += weight
    
    # Calculate capacity using the formula c = (h / (H +1)) * total_weight
        # where h = current instance and H = batch size
    h = instance_index + 1 # Capacity increases with the instance index in the batch
    capacity = int((h / (H + 1)) * total_weight)

    # Ensure the capacity is at least as large as the heaviest item, to maintain the complexity of the problem
    for value, weight in items:
        if weight > capacity:
            capacity = weight

    return items, capacity

def solve_knapsack_instances(items, capacity):
    num_items = len(items)

    # Define the problem
    prob = pulp.LpProblem("Knapsack_Problem", pulp.LpMaximize)
   
    # Decision Variables
    item_vars = [pulp.LpVariable(f'x{i}', cat='Binary') for i in range(num_items)]

    # Objective: Maximize total value
    prob += pulp.lpSum([items[i][0] * item_vars[i] for i in range(num_items)])

    # Constraint: Total weight must be <= capacity
    prob += pulp.lpSum([items[i][1] * item_vars[i] for i in range(num_items)]) <= capacity

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(timeLimit=60))

    # Check if solution is optimal
    if pulp.LpStatus[prob.status] == 'Optimal':

        # Extract the solution (binary vector indicating which items are included) and the Optimal Value (objective)
        solution = [int(var.varValue) for var in item_vars]
        return solution, pulp.value(prob.objective)
    
    return None, None # Return None if the solution is not optimal

def generate_and_solve_instances(instance_type, num_instances, num_items, R, H=100):
    instances = []
    solutions = []
    objectives = []

    # Generate instances in batches
    for i in range(num_instances):
        
        # Generate and solve the instance, regenerate until an optimal solution is found (feasible instance)
        items, capacity = generate_knapsack_instances(instance_type, num_items, i % H, R, H)

        solution, objective = None, None
        while solution is None:
            solution, objective = solve_knapsack_instances(items, capacity)
            if solution is None:
                # Regenerate the entire instance if not optimal
                items, capacity = generate_knapsack_instances(instance_type, num_items, i % H, R, H)

        # Store the instance and solution
        instances.append((items, capacity))
        solutions.append(solution)
        objectives.append(objective)

    
    # Format the data for training
    data = []
    for (items, capacity), solution, objective in zip(instances, solutions, objectives):
        # Each instance is a tuple of (items, capacity, solution)
        # items: list of (value, weight) tuples
        # capacity: integer
        # solution: binary list indicating which items to inlcude
        # objective: list of each instance's optimal value
        data.append((items, capacity, solution, objective))

    return data

# Example Usage

# if __name__ == "__main__":
    # Parameters for data generation
    # num_instances = 100  # Number of instances to generate
    # num_items = 10  # Number of items in each instance
    # R = 100  # Range for values and weights (R)
    # H = 100  # The constant H for capacity calculation

    # Generate and solve instances for Uncorrelated (UC) type
    # data_uc = generate_and_solve_instances('UC', num_instances, num_items, R, H)

    # TRAINING DATA
# data_uc_5 = generate_and_solve_instances('UC', 1000, 5, 1000, 100)
# save_data(data_uc_5, 'model_1_training_data_uc_5.pkl')

# data_uc_10 = generate_and_solve_instances('UC', 1000, 10, 1000, 100)
# save_data(data_uc_10, 'model_1_training_data_uc_10.pkl')

# data_uc_20 = generate_and_solve_instances('UC', 1000, 20, 1000, 100)
# save_data(data_uc_20, 'model_1_training_data_uc_20.pkl')

# data_uc_50 = generate_and_solve_instances('UC', 1000, 50, 1000, 100)
# save_data(data_uc_50, 'model_1_training_data_uc_50.pkl')

# data_uc_100 = generate_and_solve_instances('UC', 1000, 100, 1000, 100)
# save_data(data_uc_100, 'model_1_training_data_uc_100.pkl')

# pres_data_uc_200 = generate_and_solve_instances('UC', 100, 100, 1000, 100)
# save_data(pres_data_uc_200, 'presentation_data_uc_100.pkl')

# data_uc_200 = generate_and_solve_instances('UC', 50000, 200, 1000, 100)
# save_data(data_uc_200, 'training_data_uc_200.pkl')


# data_sc_5 = generate_and_solve_instances('SC', 50000, 5, 1000, 100)
# save_data(data_sc_5, 'model_1_training_data_sc_5.pkl')

# data_sc_10 = generate_and_solve_instances('SC', 50000, 10, 1000, 100)
# save_data(data_sc_10, 'model_1_training_data_sc_10.pkl')

# data_sc_20 = generate_and_solve_instances('SC', 50000, 20, 1000, 100)
# save_data(data_sc_20, 'model_1_training_data_sc_20.pkl')

# data_sc_50 = generate_and_solve_instances('SC', 50000, 50, 1000, 100)
# save_data(data_sc_50, 'model_1_training_data_sc_50.pkl')

# data_sc_100 = generate_and_solve_instances('SC', 50000, 100, 1000, 100)
# save_data(data_sc_100, 'model_1_training_data_sc_100.pkl')

# data_sc_200 = generate_and_solve_instances('SC', 50000, 200, 1000, 100)
# save_data(data_sc_200, 'model_1_training_data_sc_200.pkl')

# data_ss_5 = generate_and_solve_instances('SS', 50000, 5, 1000, 100)
# save_data(data_ss_5, 'model_1_training_data_ss_5.pkl')

# data_ss_10 = generate_and_solve_instances('SS', 50000, 10, 1000, 100)
# save_data(data_ss_10, 'model_1_training_data_ss_10.pkl')

# data_ss_20 = generate_and_solve_instances('SS', 50000, 20, 1000, 100)
# save_data(data_ss_20, 'model_1_training_data_ss_20.pkl')

# data_ss_50 = generate_and_solve_instances('SS', 50000, 50, 1000, 100)
# save_data(data_ss_50, 'model_1_training_data_ss_50.pkl')

# data_ss_100 = generate_and_solve_instances('SS', 50000, 100, 1000, 100)
# save_data(data_ss_100, 'model_1_training_data_ss_100.pkl')

# data_ss_200 = generate_and_solve_instances('SS', 50000, 200, 1000, 100)
# save_data(data_ss_200, 'model_1_training_data_ss_200.pkl')


# # EVALUATION DATA
# eval_data_uc_5 = generate_knapsack_instances('UC', 1000, 5, 1000, 100)
# eval_data_uc_10 = generate_knapsack_instances('UC', 1000, 10, 1000, 100)
# eval_data_uc_20 = generate_knapsack_instances('UC', 1000, 20, 1000, 100)
# eval_data_uc_50 = generate_knapsack_instances('UC', 100, 50, 1000, 100)
# eval_data_uc_100 = generate_knapsack_instances('UC', 100, 100, 1000, 100)
# eval_data_uc_200 = generate_knapsack_instances('UC', 100, 200, 1000, 100)

# eval_data_sc_5 = generate_knapsack_instances('SC', 1000, 5, 1000, 100)
# eval_data_sc_10 = generate_knapsack_instances('SC', 1000, 10, 1000, 100)
# eval_data_sc_20 = generate_knapsack_instances('SC', 1000, 20, 1000, 100)
# eval_data_sc_50 = generate_knapsack_instances('SC', 100, 50, 1000, 100)
# eval_data_sc_100 = generate_knapsack_instances('SC', 100, 100, 1000, 100)
# eval_data_sc_200 = generate_knapsack_instances('SC', 100, 200, 1000, 100)

# eval_data_ss_5 = generate_knapsack_instances('SS', 1000, 5, 1000, 100)
# eval_data_ss_10 = generate_knapsack_instances('SS', 1000, 10, 1000, 100)
# eval_data_ss_20 = generate_knapsack_instances('SS', 1000, 20, 1000, 100)
# eval_data_ss_50 = generate_knapsack_instances('SS', 100, 50, 1000, 100)
# eval_data_ss_100 = generate_knapsack_instances('SS', 100, 100, 1000, 100)
# eval_data_ss_200 = generate_knapsack_instances('SS', 100, 200, 1000, 100)


# Save data to a file

# save_data(eval_data_uc_5, 'model_1_eval_data_uc_5.pkl')
# save_data(eval_data_uc_10, 'model_1_eval_data_uc_10.pkl')
# save_data(eval_data_uc_20, 'model_1_eval_data_uc_20.pkl')
# save_data(eval_data_uc_50, 'model_1_eval_data_uc_50.pkl')
# save_data(eval_data_uc_100, 'model_1_eval_data_uc_100.pkl')
# save_data(eval_data_uc_200, 'model_1_eval_data_uc_200.pkl')

# save_data(eval_data_sc_5, 'model_1_eval_data_sc_5.pkl')
# save_data(eval_data_sc_10, 'model_1_eval_data_sc_10.pkl')
# save_data(eval_data_sc_20, 'model_1_eval_data_sc_20.pkl')
# save_data(eval_data_sc_50, 'model_1_eval_data_sc_50.pkl')
# save_data(eval_data_sc_100, 'model_1_eval_data_sc_100.pkl')
# save_data(eval_data_sc_200, 'model_1_eval_data_sc_200.pkl')

# save_data(eval_data_ss_5, 'model_1_eval_data_ss_5.pkl')
# save_data(eval_data_ss_10, 'model_1_eval_data_ss_10.pkl')
# save_data(eval_data_ss_20, 'model_1_eval_data_ss_20.pkl')
# save_data(eval_data_ss_50, 'model_1_eval_data_ss_50.pkl')
# save_data(eval_data_ss_100, 'model_1_eval_data_ss_100.pkl')
# save_data(eval_data_ss_200, 'model_1_eval_data_ss_200.pkl')

#     # Output some generated data for verification
#     print(f"Example Uncorrelated (UC) instance:\n{data_uc[0]}")

#     # You can repeat the process for Strongly Correlated (SC) and Subset Sum (SS)
#     # data_sc = generate_and_solve_instances('SC', num_instances, num_items, R, H)
#     # data_ss = generate_and_solve_instances('SS', num_instances, num_items, R, H)