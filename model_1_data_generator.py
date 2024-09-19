import pulp
import random
import numpy as np

def generate_knapsack_instances(instance_type, num_instances, num_items, value_weight_range, H=100):
    instances = []

    for _ in range(num_instances):
        items = []
        total_weight = 0

        # Generate items based on the instance type
        for _ in range(num_items):
            
            if instance_type == "UC":
                # Uncorrelated: Values and weights uniformly distributed over [1, R]
                value = random.randint(1, value_weight_range)
                weight = random.randint(1, value_weight_range)
            elif instance_type == "SC":
                # Stronlgy Correlated: value = weight + R/10
                weight = random.randint(1, value_weight_range)
                value = weight + value_weight_range // 10 
            elif instance_type == "SS":
                # Subset Sum: value equals weight
                weight = random.randint(1, value_weight_range)
                value = weight
            
            items.append((value, weight))
            total_weight += weight

        # Calculate capacity using the formula c = (h / (H + 1)) * total_weight
        h = random.randint(1, H)
        capacity = int((h / (H + 1)) * total_weight)

        # Add the instance (items and capacity) to the list
        instances.append((items, capacity))

    return instances


def solve_knapsack_instances(instances):
    solutions = []
    
    for items, capacity in instances:

        # Define the problem
        prob = pulp.LpProblem("KnapsackProblem", pulp.LpMaximize)

        # Define decision variables
        item_vars = [pulp.LpVariable(f'x{i}', cat='Binary') for i in range(len(items))]

        # Define the objective function (maximize total value)
        prob += pulp.lpSum([items[i][0] * item_vars[i] for i in range(len(items))])

        # Define the constraint (total weight <= capacity)
        prob += pulp.lpSum([items[i][1] * item_vars[i] for i in range(len(items))]) <= capacity

        # Solve the problem
        prob.solve()

        # Collect the solution
        solution = [int(item_vars[i].varValue) for i in range(len(items))]
        solutions.append(solution)

    return solutions


def generate_and_solve_instances(instance_type, num_instances, num_items, value_weight_range, H=100):
    # Generate instances
    instances = generate_knapsack_instances(instance_type, num_instances, num_items, value_weight_range, H)

    # Solve the instances to get the target solution
    solutions = solve_knapsack_instances(instances)

    # Format the data for training
    data = []
    for (items, capacity), solution in zip(instances, solutions):
        # Each instance is a tuple of (items, capacity, solution)
        # items: list of (value, weight) tuples
        # capacity: integer
        # solution: binary list indicating which items to inlcude
        data.append((items, capacity, solution))

    return data


# Example Usage

# if __name__ == "__main__":
#     # Parameters for data generation
#     num_instances = 100  # Number of instances to generate
#     num_items = 10  # Number of items in each instance
#     value_weight_range = 100  # Range for values and weights (R)
#     H = 100  # The constant H for capacity calculation

#     # Generate and solve instances for Uncorrelated (UC) type
#     data_uc = generate_and_solve_instances('UC', num_instances, num_items, value_weight_range, H)

#     # Output some generated data for verification
#     print(f"Example Uncorrelated (UC) instance:\n{data_uc[0]}")

#     # You can repeat the process for Strongly Correlated (SC) and Subset Sum (SS)
#     # data_sc = generate_and_solve_instances('SC', num_instances, num_items, value_weight_range, H)
#     # data_ss = generate_and_solve_instances('SS', num_instances, num_items, value_weight_range, H)